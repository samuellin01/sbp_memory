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
- Results with relevant content → selective omission: keep exact relevant lines verbatim, bracket out the rest with [lines N-M omitted: description]
- Totally irrelevant / outdated tool uses → one-line annotation
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

**How to rewrite — choose based on relevance:**

**If the content has relevant parts you may need again** — use **selective omission**:
  Keep exact original lines verbatim for relevant sections. Use `[lines N-M omitted: description]` markers for irrelevant sections. Be as granular as needed — you can have many alternating omit/keep sections:

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

  [lines 61-120 omitted: refund_payment, list_transactions]

  def generate_report(start_date, end_date):
      transactions = db.query(Transaction).filter(
          Transaction.date.between(start_date, end_date)
      ).all()
      return ReportBuilder(transactions).build()

  [lines 140-350 omitted: export_csv, admin helpers, CLI commands]
  ```

  Rules:
  - Preserved lines are EXACT copies from the original — never paraphrase code or structured content
  - You can have many alternating omit/keep sections (omit → keep → omit → keep → ...), not just one keep block
  - The omission markers briefly describe what was skipped
  - This applies to any tool result with relevant content: code files, diffs, error tracebacks, command output with structured data, etc.
  - ⚠️ The larger the result, the MORE important selective omission is — don't fall back to prose for big files
  - ❌ WRONG: "Viewed test/file.js (500 lines). Tests cover X, Y, Z. The edit function uses socket calls."
  - ✅ RIGHT: `[lines 1-50 omitted: ...]` then exact code then `[lines 100-500 omitted: ...]`

**If the content is totally irrelevant to the current task** — use **minimal annotation**:
  A one-line summary is fine when nothing in the result is worth preserving verbatim:
  - "Viewed /app/package.json — project config with express, jest deps. Not relevant to current task."
  - "Ran grep for 'TODO' — found 12 matches, none relevant."
  - "Build succeeded. 142 tests passed, 0 failed."
  - "Found 3 matches: src/auth.py:45, src/auth.py:112, src/middleware.py:23"

**Prioritize older, outdated tool uses** for aggressive rewriting. Prioritize using selective omission over minimal annotation.
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
- Results with relevant content → selective omission: keep exact relevant lines verbatim, bracket out the rest with [lines N-M omitted: description]
- Totally irrelevant / outdated tool uses → one-line annotation
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
