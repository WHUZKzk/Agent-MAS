"""
AutoSR — CLI entry point.

Usage:
  python -m src.main --review-index 0
  python -m src.main --review-index 0 --config configs/models.yaml
  python -m src.main --review-index 0 --checkpoint-dir data/checkpoints

Loads the ReviewConfig from data/benchmarks/bench_review.json at the given
index, then runs the SystematicReviewOrchestrator end-to-end.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
import os
import sys

from src.orchestrator import SystematicReviewOrchestrator
from src.schemas.common import PICODefinition, ReviewConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("autosr.main")


def load_review_config(bench_path: str, index: int) -> ReviewConfig:
    """Parse bench_review.json and return the ReviewConfig at position `index`."""
    with open(bench_path, encoding="utf-8") as f:
        reviews = json.load(f)

    if not isinstance(reviews, list) or index >= len(reviews):
        raise IndexError(
            f"bench_review.json has {len(reviews)} entries; "
            f"requested index {index} is out of range."
        )

    raw = reviews[index]
    pico_raw = raw["PICO"]
    return ReviewConfig(
        pmid=str(raw["PMID"]),
        title=raw["title"],
        abstract=raw["abstract"],
        pico=PICODefinition(
            P=pico_raw["P"],
            I=pico_raw["I"],
            C=pico_raw["C"],
            O=pico_raw["O"],
        ),
        # bench_review.json does not include target_characteristics / target_outcomes —
        # use sensible defaults that match the benchmark CSVs.
        target_characteristics=raw.get(
            "target_characteristics",
            ["Mean Age", "% Female", "Sample Size", "Study Duration"],
        ),
        target_outcomes=raw.get(
            "target_outcomes",
            ["Physical Activity", "MVPA", "Step Count", "BMI"],
        ),
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autosr",
        description="AutoSR — Automated Systematic Review & Meta-Analysis",
    )
    parser.add_argument(
        "--review-index", type=int, default=0,
        help="Zero-based index into bench_review.json (default: 0)",
    )
    parser.add_argument(
        "--bench-path", default="data/benchmarks/bench_review.json",
        help="Path to bench_review.json",
    )
    parser.add_argument(
        "--config", default="configs/models.yaml",
        help="Path to model registry YAML (default: configs/models.yaml)",
    )
    parser.add_argument(
        "--checkpoint-dir", default="data/checkpoints",
        help="Directory for stage checkpoints (default: data/checkpoints)",
    )
    parser.add_argument(
        "--uploads-dir", default="data/uploads",
        help="Directory containing full-text uploads (default: data/uploads)",
    )
    parser.add_argument(
        "--outputs-dir", default="data/outputs",
        help="Directory for CSV outputs (default: data/outputs)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logger.info("AutoSR starting. Review index: %d", args.review_index)
    logger.info("Bench path: %s", args.bench_path)
    logger.info("Config: %s", args.config)

    # Load review config
    try:
        review_config = load_review_config(args.bench_path, args.review_index)
    except (FileNotFoundError, IndexError, KeyError) as exc:
        logger.error("Failed to load review config: %s", exc)
        return 1

    logger.info(
        "Loaded review: PMID=%s  Title=%s",
        review_config.pmid,
        review_config.title[:80],
    )

    # Run orchestrator
    orchestrator = SystematicReviewOrchestrator(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        uploads_dir=args.uploads_dir,
        outputs_dir=args.outputs_dir,
    )

    try:
        final_state = orchestrator.run(review_config)
    except Exception as exc:
        logger.exception("Orchestrator failed: %s", exc)
        return 2

    logger.info(
        "AutoSR complete. Final stage: %s. PMID: %s",
        final_state.current_stage,
        final_state.review_config.pmid,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
