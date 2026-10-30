#!/usr/bin/env python3
"""
Compute CER, WER, and word accuracy comparing pipeline outputs against
Claude Opus 5.5 ground truth for SE_06_Q1 and SE_07_Q1.
"""
import sys
import os
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
# Look for repo root
current = Path("/mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI")
if str(current) not in sys.path:
    sys.path.insert(0, str(current))

from src.utils.text_metrics import score_transcription, cer, wer
from src.utils.linguistic_sanitizer import get_english_lexicon

def evaluate_set(dataset_name, gt_base_dir, ext_base_dir):
    print(f"\n=======================================================")
    print(f" EVALUATING: {dataset_name}")
    print(f" GT Base:  {gt_base_dir}")
    print(f" Ext Base: {ext_base_dir}")
    print(f"=======================================================")
    
    gt_pages_dir = Path(gt_base_dir) / "pages"
    if not gt_pages_dir.exists():
        print(f"Directory {gt_pages_dir} does not exist!")
        return None

    lexicon = get_english_lexicon()
    results = []

    # Iterate through scripts in gt_pages_dir
    script_dirs = sorted([d for d in gt_pages_dir.iterdir() if d.is_dir()])
    
    total_ref_chars = 0
    total_ref_words = 0
    total_s1_edits_char = 0
    total_s2_edits_char = 0
    total_s1_edits_word = 0
    total_s2_edits_word = 0

    per_script_stats = {}

    for s_dir in script_dirs:
        script_id = s_dir.name
        ckpt_dir = Path(ext_base_dir) / script_id / "checkpoints"
        if not ckpt_dir.exists():
            # Try lowercase matching
            candidates = list(Path(ext_base_dir).glob(f"*{script_id}*"))
            if candidates:
                ckpt_dir = candidates[0] / "checkpoints"

        if not ckpt_dir.exists():
            print(f"[{script_id}] Checkpoints directory not found in {ext_base_dir}")
            continue

        gt_page_txts = sorted(s_dir.glob("page_*.txt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        script_results = []

        for gt_txt in gt_page_txts:
            p_num = int(re.findall(r"\d+", gt_txt.stem)[0])
            ckpt_file = ckpt_dir / f"page_{p_num}.json"
            if not ckpt_file.exists():
                continue

            ref_text = gt_txt.read_text(encoding="utf-8")
            try:
                ckpt_data = json.loads(ckpt_file.read_text(encoding="utf-8"))
            except Exception as e:
                continue

            raw = (ckpt_data.get("stage1_transcription") or {}).get("raw_transcript") or ""
            verified = (ckpt_data.get("stage2_verification") or {}).get("verified_transcript") or ""

            s1 = score_transcription(ref_text, raw, lexicon)
            s2 = score_transcription(ref_text, verified, lexicon)

            script_results.append({
                "page": p_num,
                "ref_chars": s1.ref_chars,
                "ref_words": s1.ref_words,
                "s1_cer": s1.cer,
                "s2_cer": s2.cer,
                "s1_wer": s1.wer,
                "s2_wer": s2.wer,
                "s1_char_edits": s1.substitutions + s1.insertions + s1.deletions,
                "s2_char_edits": s2.substitutions + s2.insertions + s2.deletions,
                "s1_nonwords": s1.student_nonwords,
                "s2_nonwords_preserved": s2.nonwords_preserved,
            })

            total_ref_chars += s1.ref_chars
            total_ref_words += s1.ref_words
            total_s1_edits_char += (s1.substitutions + s1.insertions + s1.deletions)
            total_s2_edits_char += (s2.substitutions + s2.insertions + s2.deletions)

        if script_results:
            per_script_stats[script_id] = script_results
            s_chars = sum(r["ref_chars"] for r in script_results)
            s_s1_edits = sum(r["s1_char_edits"] for r in script_results)
            s_s2_edits = sum(r["s2_char_edits"] for r in script_results)
            s_cer1 = s_s1_edits / s_chars if s_chars else 0
            s_cer2 = s_s2_edits / s_chars if s_chars else 0
            mean_s1_cer = sum(r["s1_cer"] for r in script_results) / len(script_results)
            mean_s2_cer = sum(r["s2_cer"] for r in script_results) / len(script_results)
            mean_s2_wer = sum(r["s2_wer"] for r in script_results) / len(script_results)
            print(f"[{script_id}] {len(script_results):>2} pages | Micro CER: s1={s_cer1*100:.2f}%, s1+2={s_cer2*100:.2f}% | Macro CER s1+2={mean_s2_cer*100:.2f}% | Macro WER s1+2={mean_s2_wer*100:.2f}%")

    if total_ref_chars > 0:
        micro_cer_s1 = total_s1_edits_char / total_ref_chars
        micro_cer_s2 = total_s2_edits_char / total_ref_chars
        all_pages = [p for s in per_script_stats.values() for p in s]
        macro_cer_s1 = sum(p["s1_cer"] for p in all_pages) / len(all_pages)
        macro_cer_s2 = sum(p["s2_cer"] for p in all_pages) / len(all_pages)
        macro_wer_s1 = sum(p["s1_wer"] for p in all_pages) / len(all_pages)
        macro_wer_s2 = sum(p["s2_wer"] for p in all_pages) / len(all_pages)

        print(f"\n--- OVERALL SUMMARY for {dataset_name} ({len(all_pages)} pages scored across {len(per_script_stats)} scripts) ---")
        print(f"Total Reference Characters: {total_ref_chars:,}")
        print(f"Total Reference Words:      {total_ref_words:,}")
        print(f"Stage 1 Micro CER:          {micro_cer_s1*100:.2f}% (Accuracy: {(1-micro_cer_s1)*100:.2f}%)")
        print(f"Stage 1+2 Micro CER:        {micro_cer_s2*100:.2f}% (Accuracy: {(1-micro_cer_s2)*100:.2f}%)")
        print(f"Stage 1 Macro CER:          {macro_cer_s1*100:.2f}%")
        print(f"Stage 1+2 Macro CER:        {macro_cer_s2*100:.2f}%")
        print(f"Stage 1 Macro WER:          {macro_wer_s1*100:.2f}%")
        print(f"Stage 1+2 Macro WER:        {macro_wer_s2*100:.2f}%")
        return {
            "dataset": dataset_name,
            "scripts": len(per_script_stats),
            "pages": len(all_pages),
            "total_chars": total_ref_chars,
            "total_words": total_ref_words,
            "micro_cer_s1": micro_cer_s1,
            "micro_cer_s2": micro_cer_s2,
            "macro_cer_s1": macro_cer_s1,
            "macro_cer_s2": macro_cer_s2,
            "macro_wer_s1": macro_wer_s1,
            "macro_wer_s2": macro_wer_s2,
            "per_script": per_script_stats
        }
    return None

if __name__ == "__main__":
    out_dir = Path("/mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/outputs/benchmarks")
    
    r_se06 = evaluate_set(
        "SE_06_Q1",
        "/mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/data/ground_truth/transcripts/SE_06_Q1_extraction_opus_5.5",
        "/mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/outputs/extracted/english/se_06_q1"
    )

    r_se07 = evaluate_set(
        "SE_07_Q1",
        "/mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/data/ground_truth/transcripts/SE_07_Q1_extraction_opus_5.5",
        "/mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/outputs/extracted/english/se_07_q1"
    )

    summary = {"SE_06_Q1": r_se06, "SE_07_Q1": r_se07}
    summary_path = out_dir / "opus_benchmark_se06_se07.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved detailed JSON to {summary_path}")
