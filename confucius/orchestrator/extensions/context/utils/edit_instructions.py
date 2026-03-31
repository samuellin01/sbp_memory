# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Edit instruction parser and applicator for compression agent output.

Instead of requiring the compression agent to output the full rewritten content,
it outputs structured edit instructions (DELETE/REPLACE on line ranges).
Everything not mentioned is kept verbatim. This dramatically reduces the
agent's output token count — it scales with the amount of *change*, not the
amount of *retained content*.

Supported instruction format:

    DELETE <start>-<end>
    REPLACE <start>-<end>
    <replacement content, can be multiple lines>
    END_REPLACE
    SUMMARY
    <one-line summary to replace entire content>
    END_SUMMARY

Line numbers are 1-based and inclusive on both ends.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

logger: logging.Logger = logging.getLogger(__name__)

# Matches common line-number prefixes in tool output:
#   "  1|...",  " 10|...",  "100|...",  "1: ...",  " 10: ..."
_LINE_NUMBER_PREFIX: re.Pattern[str] = re.compile(
    r"^\s*(\d+)\s*[|:]\s?"
)


@dataclass
class DeleteOp:
    """Delete a range of lines."""

    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive


@dataclass
class ReplaceOp:
    """Replace a range of lines with new content."""

    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    content: str  # Replacement text


@dataclass
class SummaryOp:
    """Replace the entire content with a one-line summary."""

    content: str


EditOp = DeleteOp | ReplaceOp | SummaryOp


@dataclass
class ParseResult:
    """Result of parsing edit instructions."""

    ops: list[EditOp] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# Patterns for parsing
_DELETE_PATTERN: re.Pattern[str] = re.compile(
    r"^DELETE\s+(\d+)\s*-\s*(\d+)\s*$", re.IGNORECASE
)
_REPLACE_PATTERN: re.Pattern[str] = re.compile(
    r"^REPLACE\s+(\d+)\s*-\s*(\d+)\s*$", re.IGNORECASE
)
_END_REPLACE_PATTERN: re.Pattern[str] = re.compile(
    r"^END_REPLACE\s*$", re.IGNORECASE
)
_SUMMARY_PATTERN: re.Pattern[str] = re.compile(r"^SUMMARY\s*$", re.IGNORECASE)
_END_SUMMARY_PATTERN: re.Pattern[str] = re.compile(
    r"^END_SUMMARY\s*$", re.IGNORECASE
)


def parse_edit_instructions(raw_output: str) -> ParseResult:
    """Parse the compression agent's output into structured edit operations.

    Args:
        raw_output: Raw text output from the compression agent.

    Returns:
        ParseResult with parsed operations and any parsing errors.
    """
    result = ParseResult()
    lines = raw_output.strip().splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check for SUMMARY
        summary_match = _SUMMARY_PATTERN.match(line)
        if summary_match:
            i += 1
            summary_lines: list[str] = []
            while i < len(lines):
                if _END_SUMMARY_PATTERN.match(lines[i].strip()):
                    break
                summary_lines.append(lines[i])
                i += 1
            else:
                result.errors.append(
                    "SUMMARY block missing END_SUMMARY terminator"
                )
            result.ops.append(SummaryOp(content="\n".join(summary_lines)))
            i += 1
            continue

        # Check for DELETE
        delete_match = _DELETE_PATTERN.match(line)
        if delete_match:
            start = int(delete_match.group(1))
            end = int(delete_match.group(2))
            if start > end:
                result.errors.append(
                    f"DELETE {start}-{end}: start > end"
                )
            elif start < 1:
                result.errors.append(
                    f"DELETE {start}-{end}: line numbers must be >= 1"
                )
            else:
                result.ops.append(DeleteOp(start=start, end=end))
            i += 1
            continue

        # Check for REPLACE
        replace_match = _REPLACE_PATTERN.match(line)
        if replace_match:
            start = int(replace_match.group(1))
            end = int(replace_match.group(2))
            if start > end:
                result.errors.append(
                    f"REPLACE {start}-{end}: start > end"
                )
                i += 1
                continue
            if start < 1:
                result.errors.append(
                    f"REPLACE {start}-{end}: line numbers must be >= 1"
                )
                i += 1
                continue

            # Collect replacement content until END_REPLACE
            i += 1
            replace_lines: list[str] = []
            while i < len(lines):
                if _END_REPLACE_PATTERN.match(lines[i].strip()):
                    break
                replace_lines.append(lines[i])
                i += 1
            else:
                result.errors.append(
                    f"REPLACE {start}-{end}: missing END_REPLACE terminator"
                )
            result.ops.append(
                ReplaceOp(
                    start=start,
                    end=end,
                    content="\n".join(replace_lines),
                )
            )
            i += 1
            continue

        # Unrecognized line
        result.errors.append(f"Unrecognized instruction: {line!r}")
        i += 1

    return result


def _validate_ops(
    ops: list[EditOp], total_lines: int
) -> list[str]:
    """Validate operations don't overlap or exceed content bounds.

    Args:
        ops: List of edit operations (DELETE/REPLACE only, no SUMMARY).
        total_lines: Total number of lines in the original content.

    Returns:
        List of validation error strings (empty if valid).
    """
    errors: list[str] = []
    # Extract ranges and sort by start
    ranges: list[tuple[int, int, str]] = []
    for op in ops:
        if isinstance(op, DeleteOp):
            ranges.append((op.start, op.end, f"DELETE {op.start}-{op.end}"))
        elif isinstance(op, ReplaceOp):
            ranges.append((op.start, op.end, f"REPLACE {op.start}-{op.end}"))

    ranges.sort(key=lambda r: r[0])

    # Check bounds and overlaps
    for idx, (start, end, label) in enumerate(ranges):
        if end > total_lines:
            errors.append(
                f"{label}: end line {end} exceeds content length {total_lines}"
            )
        if idx > 0:
            prev_end = ranges[idx - 1][1]
            prev_label = ranges[idx - 1][2]
            if start <= prev_end:
                errors.append(
                    f"{label} overlaps with {prev_label}"
                )

    return errors


def apply_edit_instructions(
    original_content: str, ops: list[EditOp]
) -> str:
    """Apply parsed edit operations to the original content.

    The default for any line not covered by an operation is KEEP.

    Args:
        original_content: The original tool result content.
        ops: List of edit operations to apply.

    Returns:
        The content after applying all edit operations.

    Raises:
        ValueError: If operations overlap or are otherwise invalid.
    """
    # Handle SUMMARY — replaces everything
    summary_ops = [op for op in ops if isinstance(op, SummaryOp)]
    if summary_ops:
        return summary_ops[-1].content

    original_lines = original_content.splitlines()
    total_lines = len(original_lines)

    if total_lines == 0:
        return original_content

    # Validate
    validation_errors = _validate_ops(ops, total_lines)
    if validation_errors:
        raise ValueError(
            "Invalid edit instructions:\n" + "\n".join(validation_errors)
        )

    # Build a map of line_number -> operation
    # line numbers are 1-based
    line_ops: dict[int, DeleteOp | ReplaceOp] = {}
    for op in ops:
        if isinstance(op, (DeleteOp, ReplaceOp)):
            for line_no in range(op.start, op.end + 1):
                line_ops[line_no] = op

    # Build output
    output_parts: list[str] = []
    handled_ops: set[int] = set()  # Track ops by id() to emit replacement once

    for line_no in range(1, total_lines + 1):
        op = line_ops.get(line_no)

        if op is None:
            # KEEP — no operation covers this line
            output_parts.append(original_lines[line_no - 1])
        elif isinstance(op, DeleteOp):
            # DELETE — skip this line entirely
            continue
        elif isinstance(op, ReplaceOp):
            # REPLACE — emit replacement content only once (at start of range)
            op_id = id(op)
            if op_id not in handled_ops:
                handled_ops.add(op_id)
                output_parts.append(op.content)
            # else: skip (replacement already emitted)

    return "\n".join(output_parts)


# ── Line-number detection and remapping ──────────────────────────────────


@dataclass
class LineNumberInfo:
    """Result of detecting line numbers in content."""

    has_line_numbers: bool
    first_line_number: int = 1  # The number on the first line (usually 1)


def detect_line_numbers(content: str, sample_size: int = 10) -> LineNumberInfo:
    """Detect whether content already has line-number prefixes.

    Checks the first *sample_size* non-empty lines for a consistent
    ``N|`` or ``N:`` prefix with incrementing numbers.

    Args:
        content: The content to inspect.
        sample_size: How many lines to check (default 10).

    Returns:
        LineNumberInfo with detection result and starting line number.
    """
    lines = content.splitlines()
    if not lines:
        return LineNumberInfo(has_line_numbers=False)

    numbers: list[int] = []
    for line in lines:
        if not line.strip():
            continue
        m = _LINE_NUMBER_PREFIX.match(line)
        if m:
            numbers.append(int(m.group(1)))
        else:
            # Non-empty line without a number prefix → not numbered
            return LineNumberInfo(has_line_numbers=False)
        if len(numbers) >= sample_size:
            break

    if len(numbers) < 2:
        return LineNumberInfo(has_line_numbers=False)

    # Check that numbers are incrementing by 1
    for a, b in zip(numbers, numbers[1:]):
        if b != a + 1:
            return LineNumberInfo(has_line_numbers=False)

    return LineNumberInfo(
        has_line_numbers=True,
        first_line_number=numbers[0],
    )


def remap_ops(
    ops: list[EditOp], offset: int
) -> list[EditOp]:
    """Shift line numbers in ops so they become 1-based array indices.

    When the original content has line numbers starting at *offset*
    (e.g. 50), the compression agent's instructions reference those
    numbers. This function subtracts ``offset - 1`` so that line 50
    maps to array position 1, line 51 to 2, etc.

    Args:
        ops: Parsed edit operations with original line numbers.
        offset: The line number of the first line in the content.

    Returns:
        New list of ops with adjusted line numbers.
    """
    if offset == 1:
        return ops  # No remapping needed

    shift = offset - 1
    remapped: list[EditOp] = []
    for op in ops:
        if isinstance(op, DeleteOp):
            remapped.append(
                DeleteOp(start=op.start - shift, end=op.end - shift)
            )
        elif isinstance(op, ReplaceOp):
            remapped.append(
                ReplaceOp(
                    start=op.start - shift,
                    end=op.end - shift,
                    content=op.content,
                )
            )
        else:
            # SummaryOp — no line numbers to remap
            remapped.append(op)
    return remapped
