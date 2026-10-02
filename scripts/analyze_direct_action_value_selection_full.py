#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dap.direct_action_value_selection.full_analysis import (
    run_full_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a DAVS full run")
    parser.add_argument("run_dir")
    parser.add_argument("--output-name", default="analysis_v1")
    args = parser.parse_args()
    print(run_full_analysis(args.run_dir, args.output_name))


if __name__ == "__main__":
    main()
