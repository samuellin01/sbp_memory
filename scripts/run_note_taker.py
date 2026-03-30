#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import argparse
import asyncio
import os
import sys
from string import Template

template = Template(
    """
## Note Taking Task
Conversation Trajectory Path:
  $traj_path

Each path is a directory of conversations markdown files ordered by time.
DO NOT attempt to use bash command to view the trajectory file, you will only see truncated output.

Plz inspect the conversation in order, meanwhile immediately start taking notes; you need to output notes in the following directory:
  $note_path

AGAIN, inspect and take notes ONLY, DO NOT attempt to solve the problem, you DO NOT have access to that environment.
"""
)

from confucius.analects.note_taker.entry import CCANoteTakerEntry  # noqa: F401

from .utils import run_agent_with_prompt


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run Confucius Note Taking Agent to analyze a directory of CCA Traces"
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        required=True,
        help="Path to where the cca trajectories were stored",
    )
    parser.add_argument(
        "--note-path",
        type=str,
        required=True,
        help="Path to where the cca notes were stored",
    )

    args = parser.parse_args()
    prompt = template.substitute(traj_path=args.traj_path, note_path=args.note_path)
    os.makedirs(args.note_path)
    print(f"Output directory {args.note_path} created")
    try:
        asyncio.run(run_agent_with_prompt(prompt, entry_name="NoteTaker"))
        print("Agent completed successfully")
    except Exception as e:
        print(f"Failed to run agent: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
