#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import argparse
import asyncio
import sys
import traceback
from pathlib import Path
from string import Template

from confucius.analects.code.entry import CodeAssistEntry  # noqa: F401

from .utils import run_agent_with_prompt


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run Confucius Code Agent with a prompt from a text file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Path to the text file containing the prompt",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose logging"
    )

    # SmartContextManagementExtension configuration
    parser.add_argument(
        "--enable-smart-context",
        action="store_true",
        default=False,
        help="Enable SmartContextManagementExtension for LLM-driven context management. "
        "Trade-off: When enabled, Anthropic prompt caching is turned OFF (higher cost per "
        "request but better context control) and LLMCodingArchitectExtension is EXCLUDED "
        "(no automatic planning). When disabled (default), prompt caching reduces costs "
        "and planning is included.",
    )
    parser.add_argument(
        "--compression-threshold",
        type=int,
        default=None,
        help="Token count that triggers context optimization enforcement "
        "(maps to input_tokens_trigger in SmartContextManagementExtension). "
        "Only used when --enable-smart-context is set.",
    )
    parser.add_argument(
        "--clear-at-least",
        type=int,
        default=None,
        help="Minimum tokens that must be cleared per edit to justify the operation. "
        "Only used when --enable-smart-context is set.",
    )
    parser.add_argument(
        "--clear-at-least-tolerance",
        type=float,
        default=None,
        help="Tolerance for enforcing the clear_at_least threshold (default 0.75). "
        "Only used when --enable-smart-context is set.",
    )
    parser.add_argument(
        "--disable-enforcement",
        action="store_true",
        default=False,
        help="Disable enforcement of the clear_at_least threshold. "
        "When disabled, edits are applied immediately without accumulation. "
        "Only used when --enable-smart-context is set.",
    )
    parser.add_argument(
        "--disable-reminder",
        action="store_true",
        default=False,
        help="Disable the soft reminder SystemMessage. When disabled, enforcement happens "
        "directly at the input_tokens_trigger without prior warning. "
        "Only used when --enable-smart-context is set.",
    )
    parser.add_argument(
        "--reminder-ratio",
        type=float,
        default=None,
        help="Ratio of input_tokens_trigger at which to start showing reminders (default 0.8). "
        "Only used when --enable-smart-context is set and reminder is enabled.",
    )
    parser.add_argument(
        "--context-edit-log-dir",
        type=str,
        default=None,
        help="Directory where context_edits.jsonl should be written. "
        "Only meaningful when --enable-smart-context is set. "
        "Defaults to the current working directory when smart context is enabled.",
    )
    parser.add_argument(
        "--enable-context-usage",
        action="store_true",
        default=False,
        help="Include cumulative context usage (token count and percentage) in system_info tags. "
        "Only used when --enable-smart-context is set.",
    )
    parser.add_argument(
        "--context-window-size",
        type=int,
        default=None,
        help="Total context window size in tokens for usage percentage calculation (default 200000). "
        "Only used when --enable-context-usage is set.",
    )

    # Cache configuration (available regardless of smart context)
    parser.add_argument(
        "--cache-min-prompt-length",
        type=int,
        default=None,
        help="Minimum token length between cache breakpoints (default 10000). "
        "Applies to AnthropicPromptCaching extension.",
    )
    parser.add_argument(
        "--cache-max-num-checkpoints",
        type=int,
        default=None,
        help="Maximum number of cache breakpoints (default 4). "
        "Applies to AnthropicPromptCaching extension.",
    )

    args = parser.parse_args()

    # Read prompt from the txt file
    try:
        prompt_file = Path(args.prompt)
        if not prompt_file.exists():
            print(f"Error: File '{args.prompt}' does not exist", file=sys.stderr)
            sys.exit(1)

        if not prompt_file.suffix.lower() == ".txt":
            print(
                f"Warning: File '{args.prompt}' is not a .txt file, proceeding anyway"
            )

        problem_statement = prompt_file.read_text(encoding="utf-8").strip()

        if not problem_statement:
            print(
                "Error: The txt file is empty or contains only whitespace",
                file=sys.stderr,
            )
            sys.exit(1)

        # Create the full prompt using the task.md template with safe substitution
        template = Template(
            """## Work directory
I've uploaded a python code repository in your current directory, this will be the repository for you to investigate and make code changes.

## Problem Statement
$problem_statement

## Your Task
Can you help me implement the necessary changes to the repository so that the requirements specified in the problem statement are met?
I've already taken care of all changes to any of the test files described in the problem statement. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the $${working_dir} directory to ensure the problem statement is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the problem statement
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
3. Edit the source code of the repo to resolve the issue
4. Rerun your reproduction script and confirm that the error is fixed!
5. Think about edge cases and make sure your fix handles them as well

**Note**: this is a HARD problem, which means you need to think HARD! Your thinking should be thorough and so it's fine if it's very long.
**Note**: you are not allowed to modify project dependency files like `pyproject.toml` or `setup.py` or `requirements.txt` or `package.json`

## Exit Criteria
Please carefully follow the steps below to help review your changes.
    1. If you made any changes to your code after running the reproduction script, please run the reproduction script again.
    If the reproduction script is failing, please revisit your changes and make sure they are correct.
    If you have already removed your reproduction script, please ignore this step.

    2. Remove your reproduction script (if you haven't done so already).

    3. If you have modified any TEST files, please revert them to the state they had before you started fixing the issue.
    You can do this with `git checkout -- /path/to/test/file.py`. Use below <diff> to find the files you need to revert.

    4. Commit your change, make sure you only have one commit.
Plz make sure you commit your change at the end, otherwise I won't be able to export your change.
"""
        )

        prompt = template.substitute(problem_statement=problem_statement)

    except Exception as e:
        print(f"Error reading file '{args.prompt}': {e}", file=sys.stderr)
        print(f"Stack trace:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

    print(f"Running Confucius Code Agent with prompt from file: {args.prompt}")
    if args.enable_smart_context:
        print("SmartContextManagementExtension is ENABLED")
        print("  - Anthropic caching: ON")
        print("  - LLMCodingArchitectExtension: EXCLUDED")
        if args.compression_threshold is not None:
            print(f"  - compression_threshold: {args.compression_threshold}")
        if args.clear_at_least is not None:
            print(f"  - clear_at_least: {args.clear_at_least}")
        if args.clear_at_least_tolerance is not None:
            print(f"  - clear_at_least_tolerance: {args.clear_at_least_tolerance}")
        if args.disable_reminder:
            print("  - reminder: DISABLED")
        else:
            print("  - reminder: ENABLED")
            if args.reminder_ratio is not None:
                print(f"  - reminder_ratio: {args.reminder_ratio}")
        if args.enable_context_usage:
            print("  - context_usage: ENABLED")
            if args.context_window_size is not None:
                print(f"  - context_window_size: {args.context_window_size}")
    else:
        print("SmartContextManagementExtension is DISABLED (default)")
        print("  - Anthropic caching: ON")
        print("  - LLMCodingArchitectExtension: INCLUDED")
    
    # Print cache configuration
    if args.cache_min_prompt_length is not None or args.cache_max_num_checkpoints is not None:
        print("Cache configuration:")
        if args.cache_min_prompt_length is not None:
            print(f"  - min_prompt_length: {args.cache_min_prompt_length}")
        if args.cache_max_num_checkpoints is not None:
            print(f"  - max_num_checkpoints: {args.cache_max_num_checkpoints}")

    try:
        asyncio.run(
            run_agent_with_prompt(
                prompt,
                verbose=args.verbose,
                smart_context_enabled=args.enable_smart_context,
                smart_context_compression_threshold=args.compression_threshold,
                smart_context_clear_at_least=args.clear_at_least,
                smart_context_clear_at_least_tolerance=args.clear_at_least_tolerance,
                smart_context_enforce_clear_at_least=False if args.disable_enforcement else None,
                smart_context_reminder_enabled=False if args.disable_reminder else None,
                smart_context_reminder_ratio=args.reminder_ratio,
                cache_min_prompt_length=args.cache_min_prompt_length,
                cache_max_num_checkpoints=args.cache_max_num_checkpoints,
                context_edit_log_dir=args.context_edit_log_dir,
                smart_context_enable_context_usage=args.enable_context_usage,
                smart_context_context_window_size=args.context_window_size,
            )
        )
        print("Agent completed successfully")
    except Exception as e:
        print(f"Failed to run agent: {e}", file=sys.stderr)
        print(f"Stack trace:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
