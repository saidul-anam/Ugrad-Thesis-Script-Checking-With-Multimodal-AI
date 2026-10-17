#!/usr/bin/env python3
"""
Page-level extraction benchmark: re-run Stages 0-2b on the ground-truth pages only, then score CER.

For every script that has ground-truth pages, the stored checkpoints of its OTHER pages are copied into the
benchmark folder, so the pipeline resumes them instantly and only the ground-truth pages are transcribed
again. Stage 3/3b (script-level error extraction) is skipped here because it does not change page
transcripts. Each run writes:

  <out-dir>/<tag>/<script_id>/checkpoints/page_<n>.json    (fresh pages carry token_usage.timing)
  <out-dir>/<tag>/timing.json                              (per-stage seconds / model calls / tokens)

and then runs scripts/evaluate_transcription.py on that folder. Run it twice with the same settings to
measure the run-to-run spread before claiming any CER change.

Usage:
  python scripts/bench_pages.py --tag baseline_a
  python scripts/bench_pages.py --tag baseline_b
  python scripts/bench_pages.py --tag after --scripts SE_10_Q1_0001 SE_10_Q1_0002
  python scripts/bench_pages.py --tag no_vocab --no-prompt-vocab      # ablation: no question words in the prompts
  python scripts/bench_pages.py --tag s2_compact --stage2-compact     # ablation: shorter Stage 2 replies
"""

import os

# same CUDA allocator setting as the production runners (scripts/extract_scripts.py)
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_config
from src.core.schemas import Stage3ErrorResult
from src.engine.engine_factory import create_engine
from src.pipeline.orchestrator import ScriptCheckingPipeline
from src.utils.question_utils import load_question_for_script
from src.utils.stage_timer import merge_timings


def gt_pages(gt_dir: Path):
    out = {}
    for sdir in sorted(p for p in gt_dir.iterdir() if p.is_dir()):
        pages = sorted(int(t.stem.split("_")[1]) for t in sdir.glob("page_*.txt"))
        if pages:
            out[sdir.name] = pages
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gt-dir", default="data/ground_truth/transcripts/english_se_10_q1")
    ap.add_argument("--source-dir", default="outputs/extracted/english/se_10_q1",
                    help="stored extraction whose non-ground-truth page checkpoints are reused")
    ap.add_argument("--pdf-dir", default="data/cleaned_pdfs/english")
    ap.add_argument("--out-dir", default="outputs/benchmarks/pages")
    ap.add_argument("--config", default="configs/pipeline_config.yaml")
    ap.add_argument("--scripts", nargs="*", help="limit to these script IDs")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--stage2-compact", action="store_true",
                    help="ablation: Stage 2 replies without the unread confidence / verification_notes fields")
    ap.add_argument("--no-prompt-vocab", action="store_true",
                    help="ablation: leave the question-paper word list out of the Stage 1/2 prompts")
    args = ap.parse_args()

    pages = gt_pages(Path(args.gt_dir))
    if args.scripts:
        pages = {s: p for s, p in pages.items() if s in set(args.scripts)}
    run_dir = Path(args.out_dir) / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg.arbitration.mode = "off"
    if args.no_prompt_vocab:
        cfg.pipeline.prompt_question_vocab = False
    if args.stage2_compact:
        cfg.pipeline.stage2_compact_output = True
    engine = create_engine(cfg, force_mock=args.mock)
    pipe = ScriptCheckingPipeline(engine=engine, config=cfg)
    pipe.stage3.run = lambda *a, **k: Stage3ErrorResult()       # page transcripts do not depend on Stage 3
    question = load_question_for_script(next(iter(pages)), lang="english") if pages else None

    timings, per_script = [], {}
    for sid, gt in pages.items():
        ckpt_out = run_dir / sid / "checkpoints"
        if ckpt_out.exists():
            shutil.rmtree(ckpt_out)
        ckpt_out.mkdir(parents=True)
        src = Path(args.source_dir) / sid / "checkpoints"
        for f in src.glob("page_*.json"):
            n = int(f.stem.split("_")[1].split(".")[0])
            if n not in gt and not f.name.endswith(".reconcile.json"):
                shutil.copy(f, ckpt_out / f.name)
        res = pipe.extract_script(str(Path(args.pdf_dir) / f"{sid}.pdf"), script_id=sid, output_dir=str(run_dir),
                                  paper="english", question_input=question,
                                  pdf_samples_dir=str(run_dir / "_pages"), extract_teacher_marks=False)
        t = res.metadata.get("stage_timing") or {}
        per_script[sid] = t
        timings.append({k: v for k, v in t.items() if isinstance(v, dict)})

    summary = merge_timings(timings)
    summary["pages"] = sum(len(p) for p in pages.values())
    with open(run_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_script": per_script}, f, indent=2)
    print(json.dumps(summary, indent=1))
    shutil.rmtree(run_dir / "_pages", ignore_errors=True)

    subprocess.run([sys.executable, "scripts/evaluate_transcription.py", "--gt-dir", args.gt_dir,
                    "--extracted-dir", str(run_dir), "--tag", f"pages_{args.tag}"], check=False)


if __name__ == "__main__":
    main()
