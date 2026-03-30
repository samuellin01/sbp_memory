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
    description = f"""Edit conversation context by compressing tool results to save tokens.

Use this tool when context usage is high to optimize memory. You provide guidance on what to keep/omit for each tool result, and a compression agent handles the actual rewriting.

**Your job:** For each tool result you want to compress, provide:
- The tool_use_id
- Guidance describing what content is relevant and should be kept, and what can be omitted

**Examples of good guidance:**
- "Keep the edit function (lines ~148-154) and the delete function. Omit imports, other socket handlers."
- "Keep chat-related error entries. Omit all non-chat entries."
- "Totally irrelevant to current task — compress to a one-liner."
- "Keep the git diff hunks for assert.js. Omit the test file changes."

**Key principles:**
- **Breadth over depth**: Prefer compressing many tool results lightly over aggressively compressing a few.
- **Prioritize older, outdated tool uses** for compression. Keep recent tool uses relevant to the current task.
- **Preserve Dynamic Contexts:** Do NOT compress tool uses that load dynamic context (e.g., `skill_view`).
- **Do NOT compress** tool uses that were already rewritten in previous `{CONTEXT_EDIT_TOOL_NAME}` calls.

All tool uses not in the edit list will be kept unchanged.
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

COMPRESSION_AGENT_SYSTEM_PROMPT = """\
You are a context compression agent. Your job is to compress a tool result using selective omission.

You will receive:
1. The original tool result content
2. Guidance from the main agent about what is relevant

Your task: produce a compressed version that preserves relevant content verbatim and omits irrelevant sections.

**Rules:**

For content with relevant parts — use **selective omission**:
- Keep exact original lines VERBATIM for relevant sections — never paraphrase or summarize code
- Use `[lines N-M omitted: brief description]` markers for irrelevant sections
- Be as granular as needed — you can have many alternating omit/keep sections
- Bias toward keeping. When in doubt, keep the content. Only omit lines you are confident are irrelevant.

Example — a 350-line file where functions at lines 45-60 and 120-135 are relevant:
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

[lines 61-119 omitted: refund_payment, list_transactions]

def generate_report(start_date, end_date):
    transactions = db.query(Transaction).filter(
        Transaction.date.between(start_date, end_date)
    ).all()
    return ReportBuilder(transactions).build()

[lines 136-350 omitted: export_csv, admin helpers, CLI commands]
```

For content that is totally irrelevant — a one-line annotation is fine:
- "Viewed /app/package.json — project config with express, jest deps. Not relevant to current task."

Output ONLY the compressed content. No explanation, no preamble.
"""

COMPRESSION_AGENT_USER_PROMPT_TEMPLATE = """\
## Guidance from main agent
{guidance}

## Original tool result content ({token_count} tokens)
{content}
"""

AUTO_FORCE_MESSAGE_TEMPLATE = """⚠️ **Auto-Force Context Cleanup Triggered**

Maximum continuous edit attempts ({max_attempts}) exceeded.
Automatically forcing {forced_count} additional REWRITE operation(s) on oldest tool uses.

Auto-rewritten tool uses (oldest first):
{forced_list}

Combined with your {pending_count} pending edit(s), total savings: ~{total_savings:,} tokens

Proceeding with context optimization...
"""
