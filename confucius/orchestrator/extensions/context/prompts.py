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
- Try your best to clear at least {clear_at_least:,} tokens to justify the operation, but it's better to fall short than to delete something that will be important later.

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
- Guidance describing what content is relevant and what can be omitted, and what must be kept

**Examples of good guidance:**
- "Omit the import block (lines 1-12) and the logger/config setup (lines 13-25) — these are standard boilerplate I won't need to reference or modify. Keep everything else."
- "Omit the test setup fixtures and teardown helpers (lines 1-40), they follow standard patterns. Keep all test case functions and their assertions exactly as-is — I need to match this expected behavior."
- "Omit the unexported helper functions parseError (lines 80-95) and makeRequest (lines 96-115) — I've already read them and won't need to modify them. Keep the exported API methods and struct definitions."
- "This grep returned 50 matches across 12 files. Condense to just the file paths and line numbers — I've already identified which files to open and don't need the matched line content anymore."
- "I already applied this edit successfully and verified it works. Condense to a one-line summary of what was changed, e.g. 'Renamed Client to client in lastfm/client.go lines 15, 23, 47 (done).'"
- "This test run output shows 98 passing tests and 2 failures. Omit the individual passing test lines but note how many passed and list their names. Keep the 2 failing tests with their full tracebacks and assertion details — I need
those to debug."

**Examples of bad guidance:**
- "Keep the edit function (lines ~148-154) and the delete function. Omit the rest."
- "Keep chat-related error entries. Omit all non-chat entries."
- "Totally irrelevant to current task — compress to a one-liner."
- "Keep the git diff hunks for assert.js. Omit the test file changes."

**Key principles:**
- **str_replace() calls are very frequently called by the agent, and those require exact knowledge of the content in question, so they should not be summarized.**
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
You are a context compression agent. Your job is to output **edit instructions** that describe how to compress a tool result, based on guidance from the main agent.

You will receive:
1. The original tool result content (with line numbers)
2. Guidance from the main agent specifying what to omit or condense

**Output format:** You output edit instructions. Everything NOT mentioned in your instructions is kept verbatim. You have three instruction types:

1. `DELETE <start>-<end>` — Remove lines start through end (inclusive, 1-based).
2. `REPLACE <start>-<end>` followed by replacement text, terminated by `END_REPLACE` — Replace lines start through end with the given text.
3. `SUMMARY` followed by summary text, terminated by `END_SUMMARY` — Replace the ENTIRE content with a short summary. Use only when guidance asks for a one-liner or full condensation.

**Rules:**
- Line numbers are 1-based and inclusive on both ends.
- Ranges must not overlap.
- Only modify lines that the guidance explicitly identifies for omission or condensing.
- Everything NOT covered by an instruction is kept verbatim — you do NOT need to output kept lines.
- When replacing, use informative markers like `[lines N-M omitted: description]` so the reader knows what was removed.
- When the guidance says to "condense" rather than "omit", use REPLACE to rewrite the section into a shorter but informationally complete form.
- Never redact more broadly than the guidance asks.

**Example — guidance says "Omit the imports (lines 1-12) and the helper functions parseError and makeRequest (lines 80-115)":**

```
DELETE 1-12

REPLACE 80-115
[lines 80-115 omitted: helper functions parseError (error parsing/formatting) and makeRequest (HTTP request construction)]
END_REPLACE
```

**Example — guidance says "condense to a one-line summary of what was changed":**

```
SUMMARY
Renamed Client to client and NewClient to newClient in lastfm/client.go (lines 15, 23, 47). Edit applied and verified.
END_SUMMARY
```

**Example — guidance says "Omit the passing test details (lines 3-100) but note how many passed. Keep the 2 failures with tracebacks":**

```
REPLACE 3-100
98 tests passed: test_login, test_signup, test_validate, ...
END_REPLACE
```

**Example — guidance says "Omit the imports (lines 1-8) and the boilerplate logger setup (lines 9-15). Also condense the verbose error output (lines 60-85) to just the error message.":**

```
DELETE 1-15

REPLACE 60-85
Error: ConnectionRefusedError on port 5432 — database unreachable (3 retries exhausted)
END_REPLACE
```

Output ONLY edit instructions. No explanation, no preamble, no commentary.
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
