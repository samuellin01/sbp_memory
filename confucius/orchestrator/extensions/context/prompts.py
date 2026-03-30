# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Prompts and descriptions for smart context management extension."""

CONTEXT_EDIT_TOOL_NAME = "context_edit"

CONTEXT_MANAGEMENT_REMINDER_MESSAGE_TEMPLATE = """🚨 **URGENT: Context Usage High**: {token_count:,} tokens

Maximum compressible: ~{compressible_tokens:,} tokens (from tool uses)

**⚠️ IMPORTANT: Do NOT ignore this reminder!**

If you continue without optimizing context, you risk:
- Hitting context window limits
- Failing the entire conversation
- Losing all progress

**Action Required:**
1. **Pause current work** (if possible)
2. **Save progress to memory** (if you haven't already) - use memory tools to preserve important state, findings, decisions
3. **Use `{tool_name}` tool ASAP** to optimize memory before continuing

**🚫 DO NOT EDIT - Must Preserve:**
- Dynamic context tool uses (e.g., `skill_view`) - these load dynamic context with critical instructions
- These are reference materials you may need to consult again - rewriting them loses important details
- **Last resort exception:** If context is critically full and some skill_view results are clearly no longer relevant to the current task, they MAY be rewritten - but try all other options first

**What to rewrite:**
- Outdated exploratory work, superseded results, redundant/duplicate tool calls → rewrite to one-line annotations
- Large file reads → keep only relevant lines, bracket out the rest with [lines N-M omitted: description]
- Verbose command output → extract key result as a short summary
- Keep: Recent tool uses relevant to current task (leave unchanged)

⚠️ **Common Mistake:** Do NOT rewrite dynamic context tool uses (e.g., `skill_view`) results even if they seem verbose - you will lose reference documentation you need.

**⚠️ Important:** DO NOT try to edit tool uses that were already rewritten in previous context edits if they no longer have significant content to compress.

**Safety Constraints**:
- Try your best to clear at least {clear_at_least:,} tokens to justify the operation

⚠️ **Cache Consideration**: Clearing context invalidates cached prompt prefixes.
Make each edit worthwhile by clearing enough tokens (at least {clear_at_least:,})
and avoiding too-frequent calls.
"""


def get_context_edit_tool_description(
    keep: int,
    clear_at_least: int,
    enforce_clear_at_least: bool,
) -> str:
    """Generate tool description with config parameters."""
    description = f"""Edit conversation context by rewriting tool results to save tokens.

Use this tool when context usage is high (e.g., >50%-60% of context window) to optimize memory:
- REWRITE: Rewrite tool results on a spectrum from minimal annotation to selective omission.

**How to rewrite by tool type:**

For **file reads / code views** (str_replace_based_edit_tool, cat, etc.) — use **selective omission**:
  IMPORTANT: Do NOT summarize or paraphrase code. Keep exact original lines verbatim for relevant sections. Use `[lines N-M omitted: description]` markers for irrelevant sections.

  Example — a 350-line file where only the function at lines 45-60 is relevant to the current task:
  ```
  [lines 1-44 omitted: imports, config constants, logger setup]

  def process_payment(order_id, amount):
      tx = db.begin_transaction()
      try:
          record = PaymentRecord(order_id=order_id, amount=amount)
          tx.insert(record)
          gateway.charge(record)
          tx.commit()
      except GatewayError as e:
          tx.rollback()
          raise PaymentFailed(order_id, e)

  [lines 61-350 omitted: refund_payment, list_transactions, generate_report, export_csv, admin helpers]
  ```

  The key rule: preserved lines are EXACT copies from the original — never paraphrase code. The omission markers briefly describe what was skipped.

For **command output** (bash, terminal) — use **concise summary**:
  Extract the key result:
  - "Build succeeded. 142 tests passed, 0 failed."
  - "ERROR: ModuleNotFoundError: No module named 'requests' (full traceback omitted)"
  - "Found 3 matches: src/auth.py:45, src/auth.py:112, src/middleware.py:23"

For **least relevant / outdated tool uses** — use **minimal annotation**:
  - "Viewed /app/package.json — project config with express, jest deps. Not relevant to current task."
  - "Ran grep for 'TODO' — found 12 matches, none relevant."

**Prioritize older, outdated tool uses** for aggressive rewriting.
Keep recent tool uses that are relevant to the current task.

All tool uses not in the edit list will be kept unchanged.

**Preserve Dynamic Contexts:** Do NOT rewrite tool uses that load dynamic context (e.g., `skill_view`). These provide critical instructions and reference materials that guide task execution.

**Important:** Do NOT attempt to edit tool uses that were already rewritten in previous `{CONTEXT_EDIT_TOOL_NAME}` calls if they have minimal content remaining.
"""

    # Only include edit accumulation section if enforcement is enabled
    if enforce_clear_at_least:
        description += f"""
**Edit Accumulation:**
If your edits don't save enough tokens (< {clear_at_least:,}), they will be accumulated and you'll be asked to provide more edits. Your previous edits are saved, so you can build up to the threshold incrementally. This prevents wasting work and avoids invalidating the prompt cache prematurely.

If you provide a new edit for a tool_use_id that's already pending, it will replace the pending edit.
"""

    description += f"""
Safety:
- Try your best to clear at least {clear_at_least:,} tokens to justify the operation
- Validates that rewrites actually save tokens

⚠️ **Cache Consideration**: Clearing context invalidates cached prompt prefixes.
Make each edit worthwhile by clearing enough tokens (at least {clear_at_least:,})
and avoiding too-frequent calls.
"""
    return description


CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE = f"""\
⚠️ CRITICAL: Context limit exceeded. You MUST use {CONTEXT_EDIT_TOOL_NAME} or memory tools now - all other tools are blocked.
"""

CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE_TEMPLATE = """⚠️ CRITICAL: Context limit exceeded - {token_count:,} tokens

Maximum compressible: ~{compressible_tokens:,} tokens (from tool uses)

You MUST use `{tool_name}` or memory tools now - all other tools are blocked.

**Action Required:**
1. **Save progress to memory** (if you haven't already) - use memory tools to preserve important state, findings, decisions
2. **Use `{tool_name}` tool IMMEDIATELY** to optimize context

**🚫 DO NOT EDIT - Must Preserve:**
- Dynamic context tool uses (e.g., `skill_view`) - these load dynamic context with critical instructions
- These are reference materials you may need to consult again - rewriting them loses important details
- **Last resort exception:** If context is critically full and some skill_view results are clearly no longer relevant to the current task, they MAY be rewritten - but try all other options first

**What to rewrite:**
- Outdated exploratory work, superseded results, redundant/duplicate tool calls → minimal one-line annotations
- Large file reads → selective omission (keep relevant lines, bracket out the rest)
- Verbose command output → concise summary of key result
- Keep: Recent tool uses relevant to current task

⚠️ **Common Mistake:** Do NOT rewrite dynamic context tool uses (e.g., `skill_view`) results even if they seem verbose - you will lose reference documentation you need.

**⚠️ Important:** DO NOT try to edit tool uses that were already rewritten in previous context edits if they have minimal content remaining.

**Safety Constraints**:
- Try your best to clear at least {clear_at_least:,} tokens to justify the operation

⚠️ **Cache Consideration**: Clearing context invalidates cached prompt prefixes.
Make each edit worthwhile by clearing enough tokens (at least {clear_at_least:,})
and avoiding too-frequent calls.
"""

AUTO_FORCE_MESSAGE_TEMPLATE = """⚠️ **Auto-Force Context Cleanup Triggered**

Maximum continuous edit attempts ({max_attempts}) exceeded.
Automatically forcing {forced_count} additional REWRITE operation(s) on oldest tool uses.

Auto-rewritten tool uses (oldest first):
{forced_list}

Combined with your {pending_count} pending edit(s), total savings: ~{total_savings:,} tokens

Proceeding with context optimization...
"""
