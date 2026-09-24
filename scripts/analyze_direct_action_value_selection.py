#!/usr/bin/env python3
from __future__ import annotations

import argparse

from stage2_dynamic_budget.direct_action_value_selection.analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a DAVS minimal run")
    parser.add_argument("run_dir")
    parser.add_argument("--output-name", default="analysis_v1")
    args = parser.parse_args()
    print(run_analysis(args.run_dir, output_name=args.output_name))


if __name__ == "__main__":
    main()
