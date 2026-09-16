"""
EvidenceArbitrationGate: Stage 3b, evidence mode.

For every gate-able Stage 3 error it gathers the signals (phonetic, learned writer prior,
augmented-consensus agreement, forced choice on the line crop), fuses them into an
ambiguity score, and quantizes into GENUINE_ERROR / HANDWRITING_AMBIGUITY / UNCERTAIN.
Every record (all raw signals, verdict, crop path) is persisted so weights can be fitted and
ablated offline without GPU.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from PIL import Image

from src.core.config import ArbitrationConfig
from src.core.schemas import (
    ArbitrationCandidate,
    ArbitrationEvidence,
    ArbitrationRecord,
    LinguisticErrorItem,
    Stage3bArbitrationResult,
)
from src.pipeline.arbitration.candidate_selector import select_candidates, lexicon_candidates, levenshtein
from src.pipeline.arbitration.symbolic_evidence import align_chars, phonetic_plausibility
from src.pipeline.arbitration.writer_profile import (
    WriterProfile,
    build_writer_profile,
    add_consensus_disagreements,
    add_candidate_evidence,
    writer_prior,
    save_writer_profile,
)
from src.pipeline.arbitration.localizer import LineLocalizer, attribute_error_to_page
from src.pipeline.arbitration.consensus import ConsensusTranscriber
from src.pipeline.arbitration.forced_choice import ForcedChoiceJudge
from src.pipeline.arbitration.fusion import fuse, quantize, GENUINE, AMBIGUITY, UNCERTAIN


class EvidenceArbitrationGate:
    def __init__(
        self,
        engine,
        cfg: ArbitrationConfig,
        script_id: str,
        output_dir: str,
        page_images: List[Tuple[int, Image.Image, str]],
        page_transcripts: Dict[int, str],
        clean_image_fn: Optional[Callable[[int], Image.Image]],
        full_transcript: str,
        lexicon: Set[str],
        question_vocab: Optional[Set[str]] = None,
        language: str = "en",
        verbose: bool = True,
    ):
        self.engine = engine
        self.cfg = cfg
        self.script_id = script_id
        self.output_dir = output_dir
        self.language = language
        self.verbose = verbose
        self.page_transcripts = page_transcripts
        self.lexicon = lexicon or set()
        self.question_vocab = {w.lower() for w in (question_vocab or set())}
        self._lexicon_indices: set = set()
        self.profile: WriterProfile = build_writer_profile(script_id, full_transcript, lexicon, question_vocab)
        crop_dir = os.path.join(output_dir, "arbitration_crops") if cfg.save_crops else None
        self.localizer = LineLocalizer(
            engine=engine,
            page_images=page_images,
            page_transcripts=page_transcripts,
            clean_image_fn=clean_image_fn,
            use_bbox=cfg.use_bbox_localization,
            min_ratio=cfg.localization_min_ratio,
            search_window=cfg.projection_search_window,
            crop_pad_px=cfg.crop_pad_px,
            crop_min_height_px=cfg.crop_min_height_px,
            crop_dir=crop_dir,
        )
        self.consensus = ConsensusTranscriber(
            engine,
            n_samples=cfg.consensus_samples,
            use_sampling=cfg.consensus_use_sampling_variant,
            temperature=cfg.consensus_temperature,
            parallel_workers=getattr(cfg, "consensus_parallel_workers", 1),
        )
        self.judge = ForcedChoiceJudge(engine)
        self.records: List[ArbitrationRecord] = []
        self.cleared: List[Dict[str, Any]] = []
        self._calls_before = 0

    # ---------------------------------------------------------------- usage
    @property
    def model_calls(self) -> int:
        return self.localizer.model_calls + self.consensus.model_calls + self.judge.model_calls

    @property
    def token_usage(self) -> Dict[str, int]:
        out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for comp in (self.localizer, self.consensus, self.judge):
            for k in out:
                out[k] += comp.token_usage.get(k, 0)
        return out

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Stage 3b Gate] {msg}")

    # ------------------------------------------------------------ scoring
    def _score_candidate(self, cand: ArbitrationCandidate) -> ArbitrationRecord:
        ev = ArbitrationEvidence()
        ops = align_chars(cand.candidate_token.lower(), cand.intended_token.lower())
        ev.edit_ops = [str(o) for o in ops if o.op != "match"]

        # 1. phonetic plausibility (generic linguistic feature)
        pp = phonetic_plausibility(cand.candidate_token, cand.intended_token)
        ev.phonetic_plausibility = pp
        ev.phonetic_signal = 1.0 - pp

        # 2. learned writer prior (before this candidate contributes to the profile).
        # Check if intended_token is an established writer anchor word (demonstrated correct usage in script)
        is_writer_anchor = cand.intended_token.lower() in getattr(self.profile, "anchors", [])
        req_min = 1 if is_writer_anchor else max(1, self.cfg.writer_profile_min_count)
        wp, cnt = writer_prior(self.profile, ops, req_min)
        ev.writer_prior = wp if cnt >= req_min else None
        ev.writer_pair_count = cnt

        # 3. localization + crop evidence
        crop = None
        try:
            crop = self.localizer.locate(cand)
        except Exception as ex:
            ev.notes.append(f"localize failed: {ex}")
        if crop is not None:
            ev.localization_method = crop.method
            ev.localization_match_ratio = round(crop.match_ratio, 3)
            ev.crop_path = crop.path
            # 3a. augmented consensus
            try:
                cres = self.consensus.run(
                    crop.image,
                    crop.transcript or cand.context_sentence,
                    cand.candidate_token,
                    cand.intended_token,
                    context_sentence=cand.context_sentence,
                )
                ev.consensus_samples = [s for s in cres.get("samples", [])]
                ev.consensus_agreement_candidate = cres.get("agreement_candidate")
                ev.consensus_agreement_intended = cres.get("agreement_intended")
                ev.consensus_word_confidence = cres.get("word_confidence")
                ev.consensus_signal = cres.get("signal")
                ev.consensus_token = cres.get("consensus_token")
                ev.consensus_token_agreement = cres.get("consensus_token_agreement")
                res = cres.get("consensus")
                if res is not None and res.n >= 2:
                    add_consensus_disagreements(self.profile, res.columns, res.majority)

                # Open Multimodal Consensus:
                # If visual re-reads strongly agree (>= 60%) on a reading that differs from the candidate,
                # adopt the visual consensus reading as the intended token!
                cons_tok = cres.get("consensus_token")
                cons_agr = cres.get("consensus_token_agreement") or 0.0
                if cons_tok and cons_agr >= 0.6 and cons_tok != cand.candidate_token.lower():
                    if cand.intended_token.lower() != cons_tok:
                        cand.intended_token = cons_tok
                        ev.consensus_agreement_intended = cons_agr
                        ev.consensus_signal = min(1.0, max(0.5, 0.5 + 0.5 * cons_agr))
            except Exception as ex:
                ev.notes.append(f"consensus failed: {ex}")
            # 3b. forced choice
            try:
                fc = self.judge.judge(crop.image, cand.candidate_token, cand.intended_token,
                                      cand.context_sentence, cand.candidate_id)
                ev.forced_choice = fc.get("choice")
                ev.forced_choice_confidence = fc.get("confidence")
                ev.forced_choice_reason = fc.get("reason", "")
                ev.forced_choice_order = fc.get("order", "")
                ev.forced_choice_signal = fc.get("signal")

                # Consensus override: If visual consensus strongly agrees (>= 0.8) on intended token,
                # a contradictory forced-choice call cannot veto unanimous visual re-reads.
                if (ev.consensus_agreement_intended or 0.0) >= 0.8 and ev.forced_choice == "candidate":
                    ev.notes.append(f"Consensus agreement ({ev.consensus_agreement_intended:.2f}) on '{cand.intended_token}' overrules forced-choice '{cand.candidate_token}'")
                    ev.forced_choice_signal = 0.5  # neutralize veto
            except Exception as ex:
                ev.notes.append(f"forced choice failed: {ex}")
        else:
            ev.notes.append("no crop located; scored from model-free signals only")

        # 4. fuse + quantize
        score = fuse(ev, self.cfg.weights)
        if (ev.consensus_agreement_intended or 0.0) >= 0.8 and (ev.consensus_agreement_candidate or 0.0) <= 0.2:
            score = max(score, self.cfg.threshold_ambiguity)
        verdict = quantize(score, self.cfg.threshold_ambiguity, self.cfg.threshold_genuine)
        if is_writer_anchor and levenshtein(cand.candidate_token.lower(), cand.intended_token.lower()) == 1:
            # Benefit of the Doubt / Anchor Consistency Rule:
            # If the intended word is a demonstrated anchor word of the writer (used correctly elsewhere in the script),
            # and the discrepancy is a single-character substitution (e.g. cursive 'v' resembling 'r' in 'rillage' vs 'village'),
            # the student has demonstrated orthographic competence. Under NCTB/Cambridge assessment rules,
            # single cursive stroke deformations on known vocabulary are not penalized as genuine spelling errors.
            ev.notes.append(f"Writer anchor word '{cand.intended_token}' (mastery demonstrated elsewhere in script): Benefit of Doubt applied")
            if verdict == GENUINE:
                verdict = UNCERTAIN
        if crop is None or ev.forced_choice == "neither":
            # no ink evidence or crop failed to capture the word: benefit of the doubt / human review
            verdict = UNCERTAIN
        ev.model_calls = 0  # filled by caller from deltas
        return ArbitrationRecord(candidate=cand, evidence=ev, ambiguity_score=round(score, 4), verdict=verdict, mode="evidence")

    # ----------------------------------------------------------- arbitrate
    def arbitrate(
        self,
        errors: List[LinguisticErrorItem],
        q_no: Optional[str] = None,
        answer_text: Optional[str] = None,
    ) -> Tuple[List[LinguisticErrorItem], List[Dict[str, Any]], List[ArbitrationRecord]]:
        """
        Returns (confirmed_errors, cleared_dicts, records). An error is removed from the deductions
        if every one of its candidate pairs is HANDWRITING_AMBIGUITY or UNCERTAIN (never penalise on
        doubt); it is normalised in the transcript only for HANDWRITING_AMBIGUITY pairs.

        When `answer_text` is given and `lexicon_scan` is on, non-dictionary tokens in the answer that
        Stage 3 did not flag are gated too (Stage 3 is an LLM and misses tokens non-deterministically).
        A lexicon token confirmed GENUINE is appended to the error list as a spelling error.
        """
        errors = list(errors)
        cands = select_candidates(errors, q_no, self.cfg.max_token_edits)
        if answer_text and getattr(self.cfg, "lexicon_scan", True) and self.lexicon:
            extra_errs = lexicon_candidates(answer_text, self.lexicon, self.question_vocab, errors)
            if extra_errs:
                start = len(errors)
                errors.extend(extra_errs)
                cands.extend(select_candidates(errors, q_no, self.cfg.max_token_edits)[len(cands):])
                self._lexicon_indices = set(range(start, len(errors)))
                self._log(f"Q{q_no or '-'} lexicon scan added {len(extra_errs)} unflagged non-word(s): "
                          + ", ".join(e.erroneous_text for e in extra_errs))
            else:
                self._lexicon_indices = set()
        else:
            self._lexicon_indices = set()
        if not cands:
            return errors, [], []

        for c in cands:
            if c.page_no is None:
                c.page_no = attribute_error_to_page(c.erroneous_text, c.context_sentence, self.page_transcripts)

        q_records: List[ArbitrationRecord] = []
        for cand in cands:
            calls_before = self.model_calls
            try:
                rec = self._score_candidate(cand)
            except Exception as ex:
                rec = ArbitrationRecord(candidate=cand, evidence=ArbitrationEvidence(notes=[f"scoring failed: {ex}"]),
                                        ambiguity_score=0.5, verdict=UNCERTAIN)
            rec.evidence.model_calls = self.model_calls - calls_before
            add_candidate_evidence(self.profile, cand.candidate_token, cand.intended_token)
            q_records.append(rec)
            self._log(
                f"Q{q_no or '-'} '{cand.candidate_token}'->'{cand.intended_token}' score={rec.ambiguity_score:.2f} "
                f"{rec.verdict} [phon={_fmt(rec.evidence.phonetic_signal)} writer={_fmt(rec.evidence.writer_prior)} "
                f"cons={_fmt(rec.evidence.consensus_signal)} fc={_fmt(rec.evidence.forced_choice_signal)} "
                f"loc={rec.evidence.localization_method}]"
            )
        self.records.extend(q_records)

        by_error: Dict[int, List[ArbitrationRecord]] = {}
        for r in q_records:
            by_error.setdefault(r.candidate.error_index, []).append(r)

        confirmed: List[LinguisticErrorItem] = []
        cleared: List[Dict[str, Any]] = []
        for idx, err in enumerate(errors):
            recs = by_error.get(idx)
            if not recs:
                if idx not in self._lexicon_indices:
                    confirmed.append(err)
                continue
            verdicts = {r.verdict for r in recs}
            if GENUINE in verdicts:
                confirmed.append(err)
                continue
            # all pairs AMBIGUITY/UNCERTAIN -> drop the deduction
            for r in recs:
                if r.verdict == AMBIGUITY:
                    cleared.append({
                        "candidate": r.candidate.candidate_token,
                        "intended_word": r.candidate.intended_token,
                        "verdict": AMBIGUITY,
                        "reason": f"ambiguity={r.ambiguity_score:.2f} (loc={r.evidence.localization_method}, "
                                  f"fc={r.evidence.forced_choice}, writer_n={r.evidence.writer_pair_count})",
                        "normalize": True,
                        "candidate_id": r.candidate.candidate_id,
                    })
                else:
                    cleared.append({
                        "candidate": r.candidate.candidate_token,
                        "intended_word": r.candidate.intended_token,
                        "verdict": UNCERTAIN,
                        "reason": f"ambiguity={r.ambiguity_score:.2f}; flagged for human review",
                        "normalize": False,
                        "candidate_id": r.candidate.candidate_id,
                    })
        self.cleared.extend(cleared)
        return confirmed, cleared, q_records

    # ------------------------------------------------------------ finalize
    def finalize(self) -> Stage3bArbitrationResult:
        result = Stage3bArbitrationResult(
            mode="evidence",
            records=self.records,
            cleared=self.cleared,
            uncertain=[r for r in self.records if r.verdict == UNCERTAIN],
            total_candidates=len(self.records),
            total_model_calls=self.model_calls,
            token_usage=self.token_usage,
        )
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "stage3b_arbitration.json"), "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
            save_writer_profile(self.profile, os.path.join(self.output_dir, "writer_profile.json"))
        except Exception as ex:
            self._log(f"could not persist arbitration artifacts: {ex}")
        return result


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.2f}"
