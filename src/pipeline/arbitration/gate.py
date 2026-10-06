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
import re
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
from src.utils.strikethrough_collision_resolver import COMMON_PREPOSITIONS
from src.pipeline.arbitration.writer_profile import (
    WriterProfile,
    build_writer_profile,
    add_consensus_disagreements,
    add_candidate_evidence,
    writer_prior,
    save_writer_profile,
)
from src.pipeline.arbitration.localizer import LineLocalizer, LineCrop, attribute_error_to_page
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
        profile: Optional[WriterProfile] = None,
    ):
        self.engine = engine
        self.cfg = cfg
        self.script_id = script_id
        self.output_dir = output_dir
        self.language = language
        self.verbose = verbose
        self.page_transcripts = page_transcripts
        self.full_transcript = full_transcript or "\n\n".join(page_transcripts.values()) if page_transcripts else ""
        self.lexicon = lexicon or set()
        self.question_vocab = {w.lower() for w in (question_vocab or set())}
        self._lexicon_indices: set = set()
        self.profile: WriterProfile = profile if profile is not None else build_writer_profile(script_id, full_transcript, lexicon, question_vocab)
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
        self._total_crops_taken = 0

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

    def _detect_cursive_glyph_variant(self, c_low: str, i_low: str) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if not c_low or not i_low:
            return False, reasons

        # 1. w <-> cu split
        if c_low.replace("cu", "w") == i_low or i_low.replace("cu", "w") == c_low:
            reasons.append("asymmetric 'w'<->'cu' split")

        # 2. Palmer cursive r plateau / re shelf
        if "r" in c_low and "r" in i_low:
            c_norm = re.sub(r'r[eo]|[eo]r', 'r', c_low)
            i_norm = re.sub(r'r[eo]|[eo]r', 'r', i_low)
            if c_norm == i_norm or re.sub(r'ro|re|er|or', 'r', c_low) == i_low or re.sub(r'ro|re|er|or', 'r', i_low) == c_low:
                reasons.append("Palmer cursive 'r'<->'re' shelf")

        # 3. Looped base 's' variant
        if ("s" in c_low or c_low.endswith("s")) and c_low not in self.lexicon:
            s_suffixes = ("es", "is", "se", "so", "ss", "s")
            for suff in s_suffixes:
                if c_low.endswith(suff) and c_low[:-len(suff)] == i_low.rstrip("s"):
                    reasons.append("looped base 's' variant")
                    break

        # 4. Blind loop 'e' <-> 'c' ligature
        if levenshtein(c_low, i_low) == 1:
            ops_ec = align_chars(c_low, i_low)
            non_match = [o for o in ops_ec if o.op != "match"]
            if len(non_match) == 1 and set([non_match[0].src, non_match[0].tgt]) == {"e", "c"}:
                reasons.append("blind loop 'e'<->'c' ligature")
        if c_low not in self.lexicon and (i_low in self.lexicon or i_low in self.question_vocab):
            if ("ce" in i_low and "cee" in c_low and c_low.replace("cee", "ce", 1) == i_low) or (
                "se" in i_low and "see" in c_low and c_low.replace("see", "se", 1) == i_low
            ) or (
                c_low.endswith("ee") and i_low.endswith("e") and c_low[:-1] == i_low and any(ch in i_low[-3:] for ch in ("c", "s", "r"))
            ):
                reasons.append("blind loop ligature insertion on terminal 'c'/'s'")

        # 5. Cursive minim doubling (e.g. Palmer double-minim 'r' read as 'rr': corre -> core; letterr -> letter)
        if c_low not in self.lexicon and (i_low in self.lexicon or i_low in self.question_vocab):
            for double_char, single_char in (("rr", "r"), ("nn", "n"), ("mm", "m"), ("tt", "t")):
                if double_char in c_low and single_char in i_low and c_low.replace(double_char, single_char, 1) == i_low:
                    reasons.append(f"cursive minim doubling '{double_char}'<->'{single_char}'")
                    break

        # 6. Ascender / ligature crossbar merges (e.g. illustrates -> illustrodes; attain -> allain)
        if c_low not in self.lexicon and (i_low in self.lexicon or i_low in self.question_vocab):
            if "ll" in c_low and "tt" in i_low and c_low.replace("ll", "tt", 1) == i_low:
                reasons.append("uncrossed 'tt'<->'ll' ascender stroke")
            elif "od" in c_low and "at" in i_low and c_low.replace("od", "at", 1) == i_low:
                reasons.append("cursive ligature 'at'<->'od' loop merger")
            elif "ode" in c_low and "ate" in i_low and c_low.replace("ode", "ate", 1) == i_low:
                reasons.append("cursive ligature 'ate'<->'ode' loop merger")

        # 7. Cursive 'v' <-> 'r' ligature confusion
        if levenshtein(c_low, i_low) <= 2:
            ops_vr = align_chars(c_low, i_low)
            non_match = [o for o in ops_vr if o.op != "match"]
            swapped_chars = set()
            for o in non_match:
                swapped_chars.update([o.src, o.tgt])
            if (len(non_match) == 1 and set([non_match[0].src, non_match[0].tgt]) == {"v", "r"}) or (
                {"v", "r"}.issubset(swapped_chars) and swapped_chars.issubset({"v", "r", "u", "e", "o", "a"})
            ):
                reasons.append("cursive 'v'<->'r' ligature stroke")

        # 8. Cursive 's' <-> 'n' stroke confusion
        if levenshtein(c_low, i_low) == 1:
            ops_sn = align_chars(c_low, i_low)
            non_match = [o for o in ops_sn if o.op != "match"]
            if len(non_match) == 1 and set([non_match[0].src, non_match[0].tgt]) == {"s", "n"}:
                reasons.append("curvy 's'<->'n' stroke confusion")

        # 9. Uncrossed cursive 't' <-> 'l'
        if levenshtein(c_low, i_low) <= 2:
            ops_tl = align_chars(c_low, i_low)
            non_match = [o for o in ops_tl if o.op != "match"]
            if non_match and all(set([o.src, o.tgt]) == {"t", "l"} for o in non_match):
                reasons.append("uncrossed cursive 't'<->'l' ascender stroke")

        # 10. Cursive 'n' <-> 'r' minim/shoulder confusion
        if levenshtein(c_low, i_low) <= 2:
            ops_nr = align_chars(c_low, i_low)
            non_match = [o for o in ops_nr if o.op != "match"]
            if non_match and all(set([o.src, o.tgt]) == {"n", "r"} for o in non_match):
                reasons.append("cursive 'n'<->'r' minim/shoulder stroke confusion")

        return (len(reasons) > 0), reasons

    # ------------------------------------------------------------ scoring
    def _score_candidate(self, cand: ArbitrationCandidate) -> ArbitrationRecord:
        ev = ArbitrationEvidence()
        ops = align_chars(cand.candidate_token.lower(), cand.intended_token.lower())
        ev.edit_ops = [str(o) for o in ops if o.op != "match"]

        # 1. phonetic plausibility (generic linguistic feature)
        is_strike_suspect = (
            cand.intended_token.strip() == "[struck]"
            or cand.error_type == "strikethrough_suspect"
            or ":strike" in cand.candidate_id
        )
        if is_strike_suspect:
            ev.phonetic_plausibility = None
            ev.phonetic_signal = None  # Neutralize: strikeout suspects are visual cross-outs, not phonetic substitutions
        else:
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

        c_low = cand.candidate_token.lower()
        i_low = cand.intended_token.lower()

        # Allograph & Cursive topology checks (Palmer r, looped s, asymmetric w, blind loop e/c, minim doubling, v/r, s/n, t/l)
        is_cursive_glyph_variant, cursive_reasons = self._detect_cursive_glyph_variant(c_low, i_low)
        is_profile_allograph = getattr(self.profile, "is_writer_allograph", lambda c, i: False)(c_low, i_low)

        # Cambridge / Edexcel Test 1: Genuine Phonetic Misspelling
        # Authentic student spelling attempts at English words (e.g. 'gnowledge', 'libary', 'destruyed',
        # 'answear', 'softwor') are genuine orthographic errors when confirmed by physical ink.
        is_phonetic_error = (
            not is_strike_suspect
            and not is_cursive_glyph_variant
            and not is_profile_allograph
            and c_low not in self.lexicon
            and (i_low in self.lexicon or is_writer_anchor)
            and (ev.phonetic_plausibility or 0.0) >= 0.70
        )

        # Cambridge / Edexcel Test 2: Morphosyntactic & Grammatical Tense Switch
        # If candidate is a valid English word and the intended word is a tense/form variation
        # (e.g. 'become' -> 'became', 'produce' -> 'produced', 'has' -> 'have')
        is_grammar_tense_switch = (
            c_low in self.lexicon
            and i_low in self.lexicon
            and not is_cursive_glyph_variant
            and not is_profile_allograph
            and levenshtein(c_low, i_low) <= 2
        )

        # 3. localization + crop evidence (Budget-capped to high-value stroke ambiguity candidates)
        crop = None
        max_crops = getattr(self.cfg, "max_visual_crops", 45)
        max_q_crops = getattr(self.cfg, "max_crops_per_question", 6)
        total_crops = getattr(self, "_total_crops_taken", 0)
        cand_q = getattr(cand, "q_no", None) or getattr(cand, "question_no", None) or "ALL"
        q_crops = getattr(self, "_q_crops_taken", {}).get(cand_q, 0)

        should_attempt_crop = (
            total_crops < max_crops
            and q_crops < max_q_crops
            and not is_grammar_tense_switch
            and (
                is_strike_suspect
                or is_cursive_glyph_variant
                or is_profile_allograph
                or (c_low not in self.lexicon and levenshtein(c_low, i_low) <= 2)
                or (levenshtein(c_low, i_low) == 1 and (is_writer_anchor or c_low not in self.lexicon))
            )
        )

        if should_attempt_crop:
            try:
                crop = self.localizer.locate(cand)
                if crop is not None:
                    self._total_crops_taken = total_crops + 1
                    if not hasattr(self, "_q_crops_taken"):
                        self._q_crops_taken = {}
                    self._q_crops_taken[cand.q_no] = q_crops + 1
            except Exception as ex:
                ev.notes.append(f"localize failed: {ex}")
        else:
            ev.notes.append("visual crop skipped (budget cap or symbolic Cambridge rule)")

        if crop is not None:
            ev.localization_method = crop.method
            ev.localization_match_ratio = round(crop.match_ratio, 3)
            ev.crop_path = crop.path
            # 3a. augmented consensus
            if getattr(self.cfg, "consensus_samples", 1) > 0:
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
                    cons_tok = cres.get("consensus_token")
                    cons_agr = cres.get("consensus_token_agreement") or 0.0
                    if cons_tok and cons_agr >= 0.6 and cons_tok != cand.candidate_token.lower():
                        c_cand_low = cand.candidate_token.lower()
                        c_int_low = cand.intended_token.lower()
                        is_plausible = (
                            levenshtein(c_cand_low, cons_tok) <= 3
                            or levenshtein(c_int_low, cons_tok) <= 2
                        )
                        generic_particles = {"the", "a", "an", "is", "in", "of", "and", "to", "or", "by", "at", "it"}
                        if cons_tok in generic_particles and c_cand_low not in generic_particles:
                            is_plausible = False

                        is_valid_lexical_target = (
                            cons_tok in self.lexicon
                            or cons_tok in self.question_vocab
                            or (self.profile and getattr(self.profile, "is_writer_allograph", lambda c, i: False)(c_cand_low, cons_tok))
                        )

                        # Disputed character stroke agreement:
                        # e.g., candidate='presisely' (s) vs intended='precisely' (c), and cons_tok='presicely' (c)
                        # or candidate='strensth' (s) vs intended='strength' (g), and cons_tok='strenth' (no spurious s)
                        disputed_char_agreed = False
                        if is_plausible and c_cand_low != c_int_low:
                            ops_cand_int = align_chars(c_cand_low, c_int_low)
                            disputed_ops = [o for o in ops_cand_int if o.op != "match"]
                            cand_diff_chars = {o.src for o in disputed_ops if o.src}
                            int_diff_chars = {o.tgt for o in disputed_ops if o.tgt}
                            
                            # 1. Stroke recovery: characters in intended that candidate lacked,
                            # but consensus re-read visually observed on paper (e.g. presisely -> presicely captures 'c')
                            missing_in_cand = int_diff_chars - set(c_cand_low)
                            if missing_in_cand and any(ch in cons_tok for ch in missing_in_cand):
                                disputed_char_agreed = True
                            # 2. Spurious stroke omission: candidate had extraneous stroke that consensus dropped
                            # (e.g. strensth -> strenth dropped the spurious 's')
                            spurious_in_cand = cand_diff_chars - set(c_int_low)
                            if spurious_in_cand and not any(ch in cons_tok for ch in spurious_in_cand):
                                if levenshtein(cons_tok, c_int_low) <= levenshtein(c_cand_low, c_int_low):
                                    disputed_char_agreed = True
                            # 3. Overall edit distance improvement towards intended
                            if levenshtein(cons_tok, c_int_low) < levenshtein(c_cand_low, c_int_low):
                                disputed_char_agreed = True

                        if is_plausible and is_valid_lexical_target and cand.intended_token.lower() != cons_tok:
                            cand.intended_token = cons_tok
                            ev.consensus_agreement_intended = cons_agr
                            ev.consensus_signal = min(1.0, max(0.5, 0.5 + 0.5 * cons_agr))
                        elif is_plausible and disputed_char_agreed:
                            ev.notes.append(f"Consensus re-read '{cons_tok}' visually confirms intended character stroke in '{cand.intended_token}'")
                            ev.consensus_agreement_intended = max(ev.consensus_agreement_intended or 0.0, cons_agr)
                            ev.consensus_signal = min(1.0, max(0.70, 0.5 + 0.5 * cons_agr))
                        elif is_plausible and not is_valid_lexical_target and cand.intended_token.lower() != cons_tok:
                            ev.notes.append(f"Consensus re-read '{cons_tok}' noted: raw OCR token")
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

                if len(ev.consensus_samples) >= 3 and (ev.consensus_agreement_intended or 0.0) >= 0.8 and ev.forced_choice == "candidate":
                    ev.notes.append(f"Consensus agreement ({ev.consensus_agreement_intended:.2f}) on '{cand.intended_token}' overrules forced-choice '{cand.candidate_token}'")
                    ev.forced_choice_signal = 0.5
            except Exception as ex:
                ev.notes.append(f"forced choice failed: {ex}")
        else:
            ev.notes.append("no crop located; scored from model-free signals only")

        return self._score_candidate_with_evidence(cand, ev, crop)

    def _score_candidate_with_evidence(
        self, cand: ArbitrationCandidate, ev: ArbitrationEvidence, crop: Optional[LineCrop] = None
    ) -> ArbitrationRecord:
        c_low = cand.candidate_token.lower()
        i_low = cand.intended_token.lower()
        is_writer_anchor = i_low in getattr(self.profile, "anchors", [])
        is_strike_suspect = (
            cand.intended_token.strip() == "[struck]"
            or cand.error_type == "strikethrough_suspect"
            or ":strike" in cand.candidate_id
        )
        is_cursive_glyph_variant, cursive_reasons = self._detect_cursive_glyph_variant(c_low, i_low)
        is_profile_allograph = getattr(self.profile, "is_writer_allograph", lambda c, i: False)(c_low, i_low)
        is_phonetic_error = (
            not is_strike_suspect
            and not is_cursive_glyph_variant
            and not is_profile_allograph
            and c_low not in self.lexicon
            and (i_low in self.lexicon or is_writer_anchor)
            and (ev.phonetic_plausibility or 0.0) >= 0.70
        )
        is_grammar_tense_switch = (
            c_low in self.lexicon
            and i_low in self.lexicon
            and not is_cursive_glyph_variant
            and not is_profile_allograph
            and levenshtein(c_low, i_low) <= 2
        )

        # 4. fuse + quantize
        score = fuse(ev, self.cfg.weights)
        if (ev.consensus_agreement_intended or 0.0) >= 0.8 and (ev.consensus_agreement_candidate or 0.0) <= 0.2:
            score = max(score, self.cfg.threshold_ambiguity)

        visual_confirms_intended = (
            crop is not None
            and (
                (ev.forced_choice == "intended" and (ev.forced_choice_confidence or 0.0) >= 0.60)
                or ((ev.consensus_agreement_intended or 0.0) >= 0.70 and (ev.consensus_agreement_candidate or 0.0) <= 0.30)
            )
        )

        visual_confirms_candidate = (
            crop is not None
            and (
                (ev.forced_choice == "candidate" and (ev.forced_choice_confidence or 0.0) >= 0.70)
                or ((ev.consensus_agreement_candidate or 0.0) >= 0.70 and (ev.consensus_agreement_intended or 0.0) <= 0.20)
            )
        )

        is_cursive_glyph_variant, reasons = self._detect_cursive_glyph_variant(c_low, i_low)
        if is_cursive_glyph_variant:
            ev.notes.append(f"Cursive glyph safeguard triggered: {', '.join(reasons)}")

            if i_low in self.lexicon or i_low in self.question_vocab or is_writer_anchor:
                # Asymmetric Ambiguity Rule:
                if c_low not in self.lexicon:
                    # Candidate is an invalid non-word (e.g. 'illustrodes', 'accroding', 'rumore', 'hin', 'sourcee', 'corre'):
                    # The fact that vision model reads the continuous cursive ligature as the non-word candidate
                    # is the expected manifestation of OCR cursive ambiguity, NOT proof of a spelling fault.
                    # Elevate to forgive handwriting ambiguity regardless of visual candidate confirmation.
                    score = max(score, self.cfg.threshold_ambiguity)
                else:
                    # Candidate is ALREADY a valid English word (e.g. 'lion', 'have', 'hear'):
                    # Never mutate an already-valid word unless overwhelming visual proof of intended word exists
                    # and ink doesn't confirm candidate.
                    if not visual_confirms_candidate:
                        if (ev.consensus_agreement_intended or 0.0) >= 0.85 and ev.forced_choice == "intended":
                            score = max(score, 0.85)
                        elif score < self.cfg.threshold_genuine:
                            score = max(score, self.cfg.threshold_genuine)

        # Discovered Writer Profile Allograph check (e.g. terminal_y -> s, curvy_s -> s, cursive_vr -> v)
        is_profile_allograph = getattr(self.profile, "is_writer_allograph", lambda c, i: False)(c_low, i_low)
        if is_profile_allograph:
            if visual_confirms_candidate:
                ev.notes.append(f"Writer allograph rule '{cand.candidate_token}' -> '{cand.intended_token}' rejected: ink visually confirms candidate")
            elif is_grammar_tense_switch and not visual_confirms_intended:
                ev.notes.append(f"Writer allograph rule '{cand.candidate_token}' -> '{cand.intended_token}' rejected: grammatical tense switch requires visual proof of intended word")
            else:
                ev.notes.append(f"Writer allograph rule matched in profile: '{cand.candidate_token}' -> '{cand.intended_token}'")
                score = max(score, self.cfg.threshold_ambiguity)

        # Struck-through / aborted word & collision safeguard:
        # If student began writing a word, struck it out, and wrote the full word immediately after
        # (e.g. 'possi positively', 'grap pie-chart'), duplicate word stutter ('are are'),
        # preposition collision ('by for'), or token is enclosed in strike marks:
        is_aborted_draft = False
        words_in_ctx = re.findall(r'[A-Za-z]+(?:-[A-Za-z]+)*', cand.context_sentence.lower())
        data_viz_terms = {"graph", "grap", "chart", "pie-chart", "bar-chart", "diagram", "table", "flowchart"}
        for idx, w in enumerate(words_in_ctx[:-1]):
            if w == c_low:
                next_w = words_in_ctx[idx+1]
                # Stutter / duplicate word (e.g. 'are are')
                if w == next_w:
                    is_aborted_draft = True
                    break
                # Adjacent preposition collision (e.g. 'by for')
                if w in COMMON_PREPOSITIONS and next_w in COMMON_PREPOSITIONS and w != next_w:
                    is_aborted_draft = True
                    break
                # Partial fragment followed by word start (e.g. 'possi positively')
                if (len(c_low) >= 2 and next_w.startswith(c_low[:2])) or (len(c_low) >= 3 and levenshtein(c_low, next_w[:len(c_low)]) <= 1):
                    is_aborted_draft = True
                    break
                # False-start / abandoned draft before data visualization term (e.g. 'grap pie-chart')
                if (w in data_viz_terms or cand.intended_token.lower() in data_viz_terms) and next_w in data_viz_terms:
                    is_aborted_draft = True
                    break
        if "[struck:" in cand.context_sentence.lower():
            is_aborted_draft = True

        # Optical Strikethrough Detection on Candidate Crop:
        crop_has_strike = False
        crop_absent = (
            ev.forced_choice == "neither"
            and any(term in (ev.forced_choice_reason or "").lower() for term in ("not present", "not in image", "not visible", "missing from"))
        )
        vlm_confirms_clean_candidate = (
            ev.forced_choice == "candidate"
            and (ev.forced_choice_confidence or 0.0) >= 0.70
            and not any(term in (ev.forced_choice_reason or "").lower() for term in ("strike", "cross", "struck", "scratch", "cancel", "line through"))
        )

        if crop is not None and getattr(crop, "image", None) is not None and not crop_absent:
            if vlm_confirms_clean_candidate:
                ev.notes.append(f"Optical strikethrough skipped: VLM confirms active, unstruck handwriting for '{cand.candidate_token}'")
            else:
                try:
                    from src.pipeline.stage0_strikethrough_detector import StrikethroughDetector
                    s_det = StrikethroughDetector(min_line_width=15, max_line_height=8)
                    s_res = s_det.detect(crop.image)
                    if s_res.has_strikethrough and 1 <= s_res.region_count <= 4:
                        # Verify that detected stroke cuts through the vertical text body (20%-70% of crop height)
                        # and is not simply running along the notebook baseline ruling line (>=65%)
                        has_body_piercing_stroke = any(
                            20.0 <= getattr(reg, "y_pct", 50.0) <= 70.0
                            for reg in getattr(s_res, "regions", [])
                        ) if getattr(s_res, "regions", None) else True

                        if has_body_piercing_stroke:
                            crop_has_strike = True
                            ev.notes.append(f"Optical strikethrough confirmed on crop across '{cand.candidate_token}': {s_res.details}")
                        else:
                            ev.notes.append(f"Suppressed baseline ruling stroke across '{cand.candidate_token}' (stroke lies on notebook baseline)")
                    elif s_res.region_count > 4:
                        ev.notes.append(f"Suppressed noisy crop strokes ({s_res.region_count} regions; likely notebook ruling)")
                except Exception as ex:
                    ev.notes.append(f"crop strike detection failed: {ex}")

        if cand.error_type == "strikethrough_suspect":
            if crop_has_strike or is_aborted_draft or ev.forced_choice in ("intended", "neither"):
                ev.notes.append(f"Strikethrough suspect confirmed for '{cand.candidate_token}': Benefit of Doubt applied")
                score = max(score, 0.90)
            elif ev.forced_choice == "candidate" or not crop_has_strike:
                # Word is confirmed NOT struck through (active in text)
                if c_low in self.lexicon or is_writer_anchor:
                    ev.notes.append(f"Strikethrough suspect dismissed for '{cand.candidate_token}': word is active valid text")
                    score = max(score, 0.95)
                else:
                    score = max(score, self.cfg.threshold_ambiguity)

        # Apply aborted/struck benefit of doubt ONLY for authentic draft stutters, tagged strike suspects,
        # or optical strikes corroborated by VLM observations
        vlm_mentions_strike = any(term in (ev.forced_choice_reason or "").lower() for term in ("strike", "cross", "struck", "scratch", "cancel", "line through"))
        if is_aborted_draft or (crop_has_strike and (cand.error_type == "strikethrough_suspect" or vlm_mentions_strike)):
            ev.notes.append(f"Aborted / struck-through draft token '{cand.candidate_token}': Benefit of Doubt applied")
            score = max(score, self.cfg.threshold_ambiguity)

        # Right-Edge / Margin Truncation Safeguard:
        # If candidate token is cut off at the right margin/line end or page boundary
        from src.utils.edge_truncation_detector import is_right_edge_truncation
        if is_right_edge_truncation(
            erroneous_text=cand.candidate_token,
            suggested_correction=cand.intended_token,
            context_sentence=cand.context_sentence,
            transcript=self.full_transcript,
            lexicon=self.lexicon
        ):
            ev.notes.append(f"Right-edge / margin truncation detected for '{cand.candidate_token}' -> '{cand.intended_token}': Benefit of Doubt applied")
            score = max(score, self.cfg.threshold_ambiguity)

        has_active_exemption = (
            is_cursive_glyph_variant
            or is_profile_allograph
            or is_aborted_draft
            or crop_has_strike
            or cand.error_type == "strikethrough_suspect"
            or score >= self.cfg.threshold_ambiguity
        )

        if is_phonetic_error and not has_active_exemption:
            if visual_confirms_intended:
                ev.notes.append(f"Visual evidence confirms intended word '{cand.intended_token}': phonetic penalty waived (OCR ligature misread)")
                score = max(score, self.cfg.threshold_ambiguity)
            elif ev.forced_choice == "neither" or crop is None:
                ev.notes.append(f"Phonetic candidate '{cand.candidate_token}': visual inspection inconclusive (crop None or 'neither'), Benefit of Doubt applied")
            else:
                ev.notes.append(f"Cambridge doctrine: Phonetic misspelling '{cand.candidate_token}' -> '{cand.intended_token}' is a GENUINE spelling error")
                score = min(score, self.cfg.threshold_genuine - 0.05)

        elif is_grammar_tense_switch:
            if not visual_confirms_intended:
                ev.notes.append(f"Cambridge doctrine: Grammatical / tense switch '{cand.candidate_token}' -> '{cand.intended_token}' is a GENUINE grammatical error")
                score = min(score, self.cfg.threshold_genuine - 0.05)
            else:
                ev.notes.append(f"Visual evidence confirms intended word '{cand.intended_token}': grammatical tense switch waived (ink shows intended form)")
                score = max(score, self.cfg.threshold_ambiguity)

        verdict = quantize(score, self.cfg.threshold_ambiguity, self.cfg.threshold_genuine)
        if (
            cand.error_type == "spelling"
            and c_low not in self.lexicon
            and is_writer_anchor
            and levenshtein(cand.candidate_token.lower(), cand.intended_token.lower()) == 1
            and not visual_confirms_candidate
            and (not is_phonetic_error or visual_confirms_intended)
        ):
            # Benefit of the Doubt / Anchor Consistency Rule:
            ev.notes.append(f"Writer anchor word '{cand.intended_token}' (mastery demonstrated elsewhere in script): Benefit of Doubt applied")
            if verdict == GENUINE:
                verdict = UNCERTAIN

        if is_cursive_glyph_variant and (i_low in self.lexicon or is_writer_anchor):
            if c_low not in self.lexicon or not visual_confirms_candidate:
                if verdict == GENUINE:
                    verdict = UNCERTAIN

        if (ev.forced_choice == "neither" or (crop is None and c_low not in self.lexicon and cand.error_type == "spelling")) and verdict == GENUINE:
            ev.notes.append("Visual evidence inconclusive / no crop located for spelling non-word: Benefit of Doubt applied")
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
            # Only add candidate evidence if it's not a strike suspect and represents valid orthographic/spelling tokens
            # Grammatical tense/inflection shifts must not contaminate writer handwriting profile
            is_strike = getattr(cand, "is_strike_suspect", False) or cand.intended_token.strip() == "[struck]"
            is_grammar = "gram" in (getattr(cand, "error_type", None) or "").lower() or "synt" in (getattr(cand, "error_type", None) or "").lower()
            if not is_strike and not is_grammar:
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
            # Strikethrough / abandoned draft collision immunity
            err_clean = (err.erroneous_text or "").strip().lower()
            err_parts = err_clean.split()
            is_collision = (
                (len(err_parts) == 2 and err_parts[0] == err_parts[1])
                or (len(err_parts) == 2 and err_parts[0] in COMMON_PREPOSITIONS and err_parts[1] in COMMON_PREPOSITIONS and err_parts[0] != err_parts[1])
                or ("[struck:" in (err.context_sentence or "").lower())
            )
            if is_collision:
                repl = err.suggested_correction or ""
                if len(err_parts) == 2 and err_parts[0] == err_parts[1]:
                    repl = f"[struck: {err_parts[0]}] {err_parts[1]}"
                elif len(err_parts) == 2 and err_parts[0] in COMMON_PREPOSITIONS and err_parts[1] in COMMON_PREPOSITIONS:
                    repl = f"[struck: {err_parts[0]}] {err_parts[1]}"
                elif not repl or repl.strip().lower() == err_clean:
                    repl = err.suggested_correction or ""

                cleared.append({
                    "candidate": err.erroneous_text,
                    "intended_word": repl,
                    "context_sentence": getattr(err, "context_sentence", "") or "",
                    "erroneous_text": getattr(err, "erroneous_text", "") or "",
                    "verdict": AMBIGUITY,
                    "reason": "Strikethrough / abandoned draft collision. Zero mark deduction.",
                    "normalize": bool(repl and repl.strip().lower() != err_clean),
                    "candidate_id": f"{q_no or 'ALL'}:{idx}:0",
                })
                self._log(f"Q{q_no or '-'} '{err.erroneous_text}' cleared as Strikethrough Collision (zero deduction).")
                continue

            recs = by_error.get(idx)
            if not recs:
                if idx not in self._lexicon_indices:
                    confirmed.append(err)
                continue
            verdicts = {r.verdict for r in recs}
            if GENUINE in verdicts or UNCERTAIN in verdicts:
                confirmed.append(err)
                continue
            # all pairs AMBIGUITY -> drop the deduction
            for r in recs:
                is_strike_suspect = (r.candidate.error_type == "strikethrough_suspect")
                if r.verdict == AMBIGUITY:
                    intended = r.candidate.intended_token
                    if is_strike_suspect or intended == "[struck]":
                        intended = f"[struck: {r.candidate.candidate_token}]"
                    int_clean = (intended or "").strip().lower()
                    can_norm = bool(is_strike_suspect or int_clean in self.lexicon or int_clean in self.question_vocab)
                    cleared.append({
                        "candidate": r.candidate.candidate_token,
                        "intended_word": intended,
                        "context_sentence": r.candidate.context_sentence,
                        "erroneous_text": getattr(err, "erroneous_text", "") or "",
                        "verdict": AMBIGUITY,
                        "reason": f"ambiguity={r.ambiguity_score:.2f} (loc={r.evidence.localization_method}, "
                                  f"fc={r.evidence.forced_choice}, writer_n={r.evidence.writer_pair_count})" if not is_strike_suspect else "Strikethrough / abandoned draft verified on crop. Zero mark deduction.",
                        "normalize": can_norm,
                        "candidate_id": r.candidate.candidate_id,
                    })
                else:
                    cleared.append({
                        "candidate": r.candidate.candidate_token,
                        "intended_word": r.candidate.intended_token,
                        "context_sentence": r.candidate.context_sentence,
                        "erroneous_text": getattr(err, "erroneous_text", "") or "",
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
