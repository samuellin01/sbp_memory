# pyre-strict

"""Smart LLM-driven context management extension."""

import asyncio
import datetime
import json
import logging
from enum import Enum
from typing import Any, Awaitable, Callable, cast, override
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, PrivateAttr
from ....core.analect import AnalectRunContext, get_current_context
from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ....core import types as cf
from ....core.memory import CfMessage
from ....core.llm_manager import LLMParams
from ...exceptions import OrchestratorInterruption
from ...base import BaseOrchestrator
from ..token.estimator import TokenEstimatorExtension, MODEL_PRICING, DEFAULT_PRICING_MODEL
from ..tool_use import ToolUseExtension, ToolUseObserver
from ..caching.base import BasePromptCaching, CACHE_BREAKPOINT_KEY
from ..token.utils import calculate_image_tokens_from_dimensions, get_image_dimensions_from_block
from ...tags import Tag
from .prompts import (
    AUTO_FORCE_MESSAGE_TEMPLATE,
    COMPRESSION_AGENT_SYSTEM_PROMPT,
    COMPRESSION_AGENT_USER_PROMPT_TEMPLATE,
    CONTEXT_EDIT_TOOL_NAME,
    CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE,
    CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE_TEMPLATE,
    CONTEXT_MANAGEMENT_REMINDER_MESSAGE_TEMPLATE,
    get_context_edit_tool_description,
)
from .utils.edit_instructions import (
    apply_edit_instructions,
    parse_edit_instructions,
)
from .utils.turn_merge import merge_fully_ignored_turns

context_edit_logger = logging.getLogger("desktopenv.context_edit")

# Memory tool names used for context management exemption
MEMORY_TOOL_NAMES: list[str] = ["write_memory", "read_memory", "delete_memory", "edit_memory", "import_memory", "search_memory"]

SYS_INFO_TOOL_USE_ID_KEY = "system_info_tool_use_id"
PENDING_DATA_KEY = "pending_data"
# Marker placed in tool_use input when a tool use has been cleared by context management
CLEARED_TOOL_USE_KEY = "_context_management_cleared"
REWRITTEN_TOOL_USE_KEY = "_context_management_rewritten"


class ContextEditOperation(str, Enum):
    """Operations for context editing."""

    REWRITE = "REWRITE"


class ContextEdit(BaseModel):
    """Single context edit specification."""

    tool_use_id: str = Field(
        description="ID of tool use / function call to edit (e.g., 'toolu_bdrk_abc123', 'call_xyz123'), as shown in the <system_info> tag AFTER each function call result"
    )

    operation: ContextEditOperation = Field(
        description=(
            "REWRITE: Compress tool result to save tokens. "
            "A compression agent will handle the actual rewriting based on your guidance."
        )
    )

    guidance: str = Field(
        default="",
        description="Describe what content is relevant and should be kept, and what can be omitted. "
        "Examples: 'Keep the edit function and delete function, omit everything else', "
        "'Totally irrelevant — compress to one-liner', "
        "'Keep chat-related error entries, omit all non-chat entries'",
    )

    tool_use_input_replacement: dict[str, Any] | None = Field(
        default=None,
        description="Optional: replacement for verbose tool input (rarely needed since inputs are usually small)",
    )

    # Filled by compression agent, not by the main agent
    tool_result_content_replacement: str | None = Field(
        default=None,
        description="Compressed tool result content. Populated by the compression agent — do not fill this yourself.",
        exclude=True,  # Exclude from JSON schema shown to the main agent
    )

    reason: str | None = Field(
        default=None, description="Why this edit is needed (for debugging/logging)"
    )


class ContextEditInput(BaseModel):
    """Input schema for context_edit tool."""

    edits: list[ContextEdit] = Field(description="List of context edits to apply")


class RejectedEdit(BaseModel):
    """An individual edit that was rejected during validation."""

    edit: ContextEdit
    reason: str


class ReplacementInfo(BaseModel):
    """Information about an edit that replaced a pending edit."""

    tool_use_id: str
    old_operation: ContextEditOperation
    new_operation: ContextEditOperation
    old_savings: int
    new_savings: int
    delta: int


class ValidationResult(BaseModel):
    """Result of edit validation with clear success/failure semantics.

    Attributes:
        is_safe: Whether validation passed (True) or failed (False)
        is_pending: Whether edits are valid but need more for savings threshold (True) or not (False)
        valid_edits: Edits that passed validation and can be applied
        rejected_edits: Individual edits that were rejected (with reasons)
        failure_reason: Global validation failure reason (when is_safe=False, all edits rejected)
        pending_reason: Why edits are pending (when is_pending=True)
        estimated_savings: Estimated token savings from the edits
        edit_savings_map: Map of tool_use_id to individual savings
        replacement_info: Information about edits that replaced pending edits
    """

    is_safe: bool
    is_pending: bool = False
    valid_edits: list[ContextEdit] = Field(default_factory=list)
    rejected_edits: list[RejectedEdit] = Field(default_factory=list)
    failure_reason: str | None = None
    pending_reason: str | None = None
    estimated_savings: int = 0
    edit_savings_map: dict[str, int] = Field(default_factory=dict)
    replacement_info: list[ReplacementInfo] = Field(default_factory=list)


class EditStats(BaseModel):
    """Statistics about applied edits."""

    rewritten_count: int
    kept_count: int
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float


class _ToolUseMessages(BaseModel):
    """Message references for a single tool use (CfMessage references)."""

    # Store direct CfMessage references for in-place edits
    tool_use: CfMessage | None = None
    tool_result: CfMessage | None = None
    tool_use_info: CfMessage | None = None

    # Convenience methods to check existence without triggering lookup
    def has_tool_use(self) -> bool:
        """Check if tool_use is set without triggering message lookup."""
        return self.tool_use is not None

    def has_tool_result(self) -> bool:
        """Check if tool_result is set without triggering message lookup."""
        return self.tool_result is not None

    def has_tool_use_info(self) -> bool:
        """Check if tool_use_info is set without triggering message lookup."""
        return self.tool_use_info is not None


_ToolUseMessageMap = dict[str, _ToolUseMessages]


class _PendingData(BaseModel):
    edits: list[ContextEdit] = Field(default_factory=list)
    estimated_savings: int = 0
    edit_savings_map: dict[str, int] = Field(default_factory=dict)

    def clear(self) -> None:
        """Clear all pending data."""
        self.edits = []
        self.estimated_savings = 0
        self.edit_savings_map = {}


class _ToolUseResult(BaseModel):
    tool_use: ant.MessageContentToolUse
    tool_result: ant.MessageContentToolResult


class SmartContextManagementExtension(
    TokenEstimatorExtension, ToolUseExtension, ToolUseObserver
):
    """
    Smart LLM-driven context management via direct memory manipulation.

    This extension provides the context_edit tool that allows the LLM to
    intelligently manage conversation context by selectively removing or
    compacting tool uses based on their relevance to the current task.

    Example usage in an Entry:

        extensions=[
            SmartContextManagementExtension(
                keep=10,  # Keep at least 10 tool uses
                clear_at_least=75000,  # Clear 75K+ tokens per edit
                input_tokens_trigger=150000,  # Trigger enforcement at 150K tokens
            ),
        ]

    Cache Consideration:
    Prompt caching allows reusing prompt prefixes across requests.
    However, context editing invalidates the cache because message history changes.

    The clear_at_least threshold ensures edits save enough tokens to justify
    the cache invalidation cost. Set this based on your usage patterns:
    - High frequency, long sessions: Set higher (100K+)
    - Low frequency, short sessions: Set lower (25K-50K)
    """

    included_in_system_prompt: bool = False

    # Configuration fields
    keep: int = Field(
        default=0,
        description="Minimum number of tool uses to retain",
    )
    clear_at_least: int = Field(
        default=50000,
        description="Minimum tokens that must be cleared per edit to justify the operation",
    )
    enforce_clear_at_least: bool = Field(
        default=False,
        description="Whether to enforce the clear_at_least threshold",
    )
    enforce_clear_at_least_tolerance: float = Field(
        default=0.75,
        description="Tolerance for enforcing the clear_at_least threshold, to account for token estimation errors",
    )
    input_tokens_trigger: int = Field(
        default=100000,
        description="Input token count that triggers context optimization enforcement",
    )
    reminder_enabled: bool = Field(
        default=True,
        description="Whether to inject the soft reminder SystemMessage when approaching the trigger threshold",
    )
    reminder_ratio: float = Field(
        default=0.8,
        description="Ratio of input_tokens_trigger at which to start showing reminders (only used when reminder_enabled=True)",
    )
    reminder_message_template: str = Field(
        default=CONTEXT_MANAGEMENT_REMINDER_MESSAGE_TEMPLATE,
        description="The reminder message to inject when triggered",
    )
    enforcement_message_template: str = Field(
        default=CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE_TEMPLATE,
        description="Enforcement message template for context management",
    )
    remove_context_edit_history: bool = Field(
        default=True,
        description="Remove context_edit tool messages from history after execution",
    )
    max_continuous_edits: int = Field(
        default=3,
        description="Maximum number of continuous context_edit attempts before auto-forcing additional REWRITE edits on oldest tool uses",
    )
    merge_fully_ignored_turns_enabled: bool = Field(
        default=True,
        description="Merge consecutive fully-ignored turns into single system message to save tokens",
    )
    log_dir: str | None = Field(
        default="/tmp",
        description="Directory to write context_edits.json log file. Defaults to /tmp to avoid the agent accidentally git-adding or deleting it.",
    )
    context_window_size: int = Field(
        default=200000,
        description="Total context window size in tokens, used for context usage percentage in system_info",
    )
    enable_context_usage: bool = Field(
        default=False,
        description="Whether to include cumulative context usage in system_info tags after tool use",
    )
    compression_agent_enabled: bool = Field(
        default=False,
        description="Whether to use a separate compression agent for rewriting tool results. "
        "When enabled, the main agent provides guidance and a compression agent handles the actual rewriting.",
    )
    compression_agent_model: str | None = Field(
        default=None,
        description="Model ID for the compression agent. If None, uses the same model as the main agent.",
    )
    compression_agent_max_tokens: int = Field(
        default=16384,
        description="Max tokens for compression agent responses.",
    )
    training_data_dir: str | None = Field(
        default=None,
        description="Directory to write training data JSONL files (context_snapshots.jsonl, "
        "tool_use_registry.jsonl, edit_outcomes.jsonl). If None, no training data collected.",
    )
    _tool_use_message_map: _ToolUseMessageMap = PrivateAttr(default_factory=dict)
    _ignored_tool_use_ids: set[str] = PrivateAttr(default_factory=set)
    _last_added_tool_use_id: str | None = PrivateAttr(default=None)
    _context_edit_tool_uses: dict[str, _ToolUseResult] = PrivateAttr(
        default_factory=dict
    )
    _compressible_tokens: int | None = PrivateAttr(default=None)
    _continuous_edit_count: int = PrivateAttr(default=0)
    _enforce_mode: bool = PrivateAttr(default=False)
    _current_model: str | None = PrivateAttr(default=None)
    _compression_agent_usage: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    # Training data collection state
    _turn_counter: int = PrivateAttr(default=0)
    # Registry of all tool uses: tool_use_id -> metadata
    _tool_use_registry: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    # Track which tool_use_ids were edited and how
    _edit_history: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def _get_tool_use_messages(self, tool_use_id: str) -> _ToolUseMessages:
        """Get the messages for a single tool use."""
        return self._tool_use_message_map.setdefault(tool_use_id, _ToolUseMessages())

    def _log_context_edit(self, data: dict) -> None:
        """Log a context edit event at INFO level."""
        summary = data.get("event", "context_edit")
        details = {k: v for k, v in data.items() if k != "event"}
        context_edit_logger.info("%s: %s", summary, details)

    def _write_context_edit_json(self, data: dict) -> None:
        """Write a consolidated context edit record to context_edits.json."""
        if self.log_dir is None:
            return
        import os
        log_path = os.path.join(self.log_dir, "context_edits.json")
        record = {"timestamp": datetime.datetime.utcnow().isoformat(), **data}
        try:
            entries: list[dict[str, Any]] = []
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            entries.append(record)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, default=str)
        except Exception as exc:
            context_edit_logger.warning("Failed to write context_edits.json: %s", exc)

    def _log_training_data(self, filename: str, data: dict) -> None:
        """Write a training data record to a JSONL file in training_data_dir."""
        if self.training_data_dir is None:
            return
        import os
        os.makedirs(self.training_data_dir, exist_ok=True)
        log_path = os.path.join(self.training_data_dir, filename)
        record = {"timestamp": datetime.datetime.utcnow().isoformat(), **data}
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            context_edit_logger.warning("Failed to write %s: %s", filename, exc)

    def _extract_content_preview(self, content: Any) -> str:
        """Extract a text representation from tool use input or result content."""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return json.dumps(content, default=str)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
            return " ".join(parts)
        return str(content)

    def _extract_tool_use_input_and_result(
        self, tool_use_id: str, tool_use_msg: CfMessage | None, tool_result_msg: CfMessage | None
    ) -> tuple[str, str]:
        """Extract input preview and result preview for a tool use."""
        input_preview = ""
        result_preview = ""
        if tool_use_msg is not None and isinstance(tool_use_msg.content, list):
            for item in tool_use_msg.content:
                try:
                    tu = ant.MessageContentToolUse.model_validate(item)
                    if tu.id == tool_use_id:
                        input_preview = self._extract_content_preview(tu.input)
                        break
                except Exception:
                    continue
        if tool_result_msg is not None and isinstance(tool_result_msg.content, list):
            for item in tool_result_msg.content:
                try:
                    tr = ant.MessageContentToolResult.model_validate(item)
                    if tr.tool_use_id == tool_use_id:
                        result_preview = self._extract_content_preview(tr.content)
                        break
                except Exception:
                    continue
        return input_preview, result_preview

    @property
    def _pending_data(self) -> _PendingData:
        """Get the pending data for the current context edit session."""
        context = get_current_context()
        return cast(
            _PendingData,
            context.session_storage[self.__class__.__name__].setdefault(
                PENDING_DATA_KEY, _PendingData()
            ),
        )

    @property
    def practical_clear_at_least(self) -> int:
        """
        Get the practical minimum token threshold with tolerance applied.

        Takes the smaller of _compressible_tokens and clear_at_least, then applies
        tolerance to account for token estimation errors. This ensures we don't
        require clearing more tokens than are actually available while being
        forgiving about estimation errors.
        """
        # Start with clear_at_least as the base
        effective_threshold = self.clear_at_least

        # If compressible tokens is known and smaller, use that instead
        if (
            self._compressible_tokens is not None
            and self._compressible_tokens < self.clear_at_least
        ):
            effective_threshold = self._compressible_tokens

        # Apply tolerance to the effective threshold
        return int(effective_threshold * self.enforce_clear_at_least_tolerance)

    @property
    def enforced_input_tokens_trigger(self) -> int:
        """Get the enforced input tokens trigger (now just returns input_tokens_trigger directly)."""
        return self.input_tokens_trigger

    def _is_message_fully_processed(self, msg: CfMessage) -> bool:
        """Check if ALL tool_use_ids in this message have been processed (cleared or rewritten)."""
        tool_use_ids = self._extract_tool_use_ids_from_message(msg)
        if not tool_use_ids:
            return False
        return all(tid in self._ignored_tool_use_ids for tid in tool_use_ids)

    async def _calculate_compressible_tokens(self, context: AnalectRunContext) -> int:
        """
        Calculate total tokens from all tracked tool use messages (roofline).

        This represents the maximum possible token savings if all tool uses
        were rewritten to minimal content.

        Args:
            context: The analect run context.

        Returns:
            Total tokens from all non-processed tool use, tool result, and tool_use_info messages.
        """
        self._build_tool_use_message_map(context)
        total = 0
        for tool_use_id, tool_use_msgs in self._tool_use_message_map.items():
            tool_use = tool_use_msgs.tool_use
            if tool_use is not None and tool_use_id not in self._ignored_tool_use_ids:
                messages_to_count = [tool_use]
                if tool_use_msgs.tool_result is not None:
                    messages_to_count.append(tool_use_msgs.tool_result)
                if tool_use_msgs.tool_use_info is not None:
                    messages_to_count.append(tool_use_msgs.tool_use_info)

                if messages_to_count:
                    token_lengths = await self.get_prompt_token_lengths(
                        messages_to_count
                    )
                    total += sum(token_lengths)
        return total

    def _get_context_edit_schema(self) -> dict[str, Any]:
        """Get JSON schema for ContextEditInput, excluding internal fields."""
        schema = ContextEditInput.model_json_schema()
        # Remove tool_result_content_replacement from ContextEdit schema —
        # it's populated by the compression agent, not by the main agent
        if "$defs" in schema and "ContextEdit" in schema["$defs"]:
            props = schema["$defs"]["ContextEdit"].get("properties", {})
            props.pop("tool_result_content_replacement", None)
        return schema

    @override
    @property
    async def tools(self) -> list[ant.ToolLike]:
        """Return context_edit tool."""
        return [
            ant.Tool(
                name=CONTEXT_EDIT_TOOL_NAME,
                description=get_context_edit_tool_description(
                    keep=self.keep,
                    clear_at_least=self.clear_at_least,
                    enforce_clear_at_least=self.enforce_clear_at_least,
                ),
                input_schema=self._get_context_edit_schema(),
            )
        ]

    @override
    async def on_invoke_llm_with_params(
        self,
        messages: list[BaseMessage],
        llm_params: LLMParams,
        context: AnalectRunContext,
    ) -> tuple[list[BaseMessage], LLMParams]:
        # Call super to allow other extensions to process
        (
            messages,
            llm_params,
        ) = await super().on_invoke_llm_with_params(messages, llm_params, context)

        # Capture the current model for compression agent fallback
        if llm_params.model is not None:
            self._current_model = llm_params.model

        prompt_lengths = await self.get_prompt_token_lengths(
            messages, tools=(llm_params.additional_kwargs or {}).get("tools")
        )
        total_length = sum(prompt_lengths)

        # Training data: per-turn context snapshot
        self._turn_counter += 1
        if self.training_data_dir is not None:
            self._build_tool_use_message_map(context)
            tool_use_summary = []
            for tuid, msgs in self._tool_use_message_map.items():
                if msgs.tool_use is None:
                    continue
                tool_name = self._extract_tool_name(tuid, msgs.tool_use)
                if tool_name == CONTEXT_EDIT_TOOL_NAME:
                    continue
                registry_entry = self._tool_use_registry.get(tuid, {})
                tool_use_summary.append({
                    "id": tuid,
                    "name": tool_name,
                    "created_at_turn": registry_entry.get("created_at_turn", -1),
                    "age_turns": self._turn_counter - registry_entry.get("created_at_turn", self._turn_counter),
                    "input_tokens": registry_entry.get("input_tokens", 0),
                    "result_tokens": registry_entry.get("result_tokens", 0),
                    "image_count": registry_entry.get("image_count", 0),
                    "image_tokens": registry_entry.get("image_tokens", 0),
                    "rewritten": tuid in self._ignored_tool_use_ids,
                })
            cache_read = None
            last_prompt = self.get_last_prompt_token_length()
            if last_prompt is not None and last_prompt > 0:
                state = self._get_state(context)
                cache_read_tokens = (
                    state.last_prompt_token_length
                    if hasattr(state, "last_prompt_token_length")
                    else None
                )
            self._log_training_data("context_snapshots.jsonl", {
                "event": "context_snapshot",
                "turn_number": self._turn_counter,
                "total_tokens": total_length,
                "context_fill_ratio": round(total_length / self.context_window_size, 4),
                "num_tool_uses": len(tool_use_summary),
                "num_rewritten": sum(1 for t in tool_use_summary if t["rewritten"]),
                "compressible_tokens": self._compressible_tokens,
                "enforce_mode": self._enforce_mode,
                "tool_use_summary": tool_use_summary,
            })

        # Calculate reminder and enforcement triggers
        reminder_trigger = int(self.input_tokens_trigger * self.reminder_ratio) if self.reminder_enabled else self.input_tokens_trigger
        enforcement_trigger = self.input_tokens_trigger

        # Calculate _compressible_tokens when above the minimum trigger threshold
        if total_length >= min(reminder_trigger, enforcement_trigger):
            self._compressible_tokens = await self._calculate_compressible_tokens(
                context
            )

        # Inject reminder message if enabled and above reminder threshold
        if self.reminder_enabled and total_length >= reminder_trigger:
            messages.append(
                SystemMessage(
                    content=self.reminder_message_template.format(
                        token_count=total_length,
                        compressible_tokens=self._compressible_tokens,
                        tool_name=CONTEXT_EDIT_TOOL_NAME,
                        keep=self.keep,
                        clear_at_least=self.practical_clear_at_least,
                    )
                )
            )

        # Inject enforcement message if in enforce mode
        if self._enforce_mode:
            # _compressible_tokens may be None if the reminder threshold was not crossed
            # (e.g. enforcement triggered before the reminder ratio), so calculate it here.
            if self._compressible_tokens is None:
                self._compressible_tokens = await self._calculate_compressible_tokens(
                    context
                )
            messages.append(
                SystemMessage(
                    content=self.enforcement_message_template.format(
                        token_count=total_length,
                        compressible_tokens=self._compressible_tokens,
                        tool_name=CONTEXT_EDIT_TOOL_NAME,
                        keep=self.keep,
                        clear_at_least=self.practical_clear_at_least,
                    )
                )
            )

        return messages, llm_params

    def _extract_tool_use_ids_from_message(self, msg: CfMessage) -> set[str]:
        """
        Extract all tool_use_ids from a message.

        Handles three types of messages:
        - AI messages with tool uses (extracts tool_use.id)
        - HUMAN messages with tool results (extracts tool_result.tool_use_id)
        - SYS messages with tool_use_id in additional_kwargs

        Args:
            msg: The message to extract tool_use_ids from

        Returns:
            Set of tool_use_ids found in the message
        """
        tool_use_ids = set()

        # Check SYS message with tool_use_id in additional_kwargs
        if msg.type == cf.MessageType.SYS:
            tool_use_id = msg.additional_kwargs.get(SYS_INFO_TOOL_USE_ID_KEY)
            if tool_use_id:
                tool_use_ids.add(tool_use_id)
            return tool_use_ids

        # Check message content blocks
        if not isinstance(msg.content, list):
            return tool_use_ids

        for content_item in msg.content:
            # Extract from AI messages (tool uses)
            if msg.type == cf.MessageType.AI:
                try:
                    tool_use = ant.MessageContentToolUse.model_validate(content_item)
                    tool_use_ids.add(tool_use.id)
                except Exception:
                    # Not a tool use, skip to next item
                    continue

            # Extract from HUMAN messages (tool results)
            elif msg.type == cf.MessageType.HUMAN:
                try:
                    tool_result = ant.MessageContentToolResult.model_validate(
                        content_item
                    )
                    tool_use_ids.add(tool_result.tool_use_id)
                except Exception:
                    # Not a tool result, skip to next item
                    continue

        return tool_use_ids

    def _build_tool_use_message_map(self, context: AnalectRunContext) -> None:
        """
        Build map of tool_use_id to CfMessage references.

        This allows fast lookup when applying edits.
        """
        self._tool_use_message_map.clear()
        self._ignored_tool_use_ids.clear()  # Reset and rebuild from message content
        messages = context.memory_manager.get_session_memory().messages

        for msg in messages:
            tool_use_ids = self._extract_tool_use_ids_from_message(msg)

            for tool_use_id in tool_use_ids:
                # Store message reference based on message type
                if msg.type == cf.MessageType.AI:
                    self._get_tool_use_messages(tool_use_id).tool_use = msg
                    # Check if this specific tool_use_id has been cleared
                    if self._is_tool_use_content_cleared(msg, tool_use_id):
                        self._ignored_tool_use_ids.add(tool_use_id)
                elif msg.type == cf.MessageType.HUMAN:
                    self._get_tool_use_messages(tool_use_id).tool_result = msg
                elif msg.type == cf.MessageType.SYS:
                    self._get_tool_use_messages(tool_use_id).tool_use_info = msg

    def _is_tool_use_content_cleared(self, msg: CfMessage, tool_use_id: str) -> bool:
        """Check if a specific tool_use_id within a message has been cleared."""
        if not isinstance(msg.content, list):
            return False
        for item in msg.content:
            try:
                tool_use = ant.MessageContentToolUse.model_validate(item)
                if tool_use.id == tool_use_id:
                    return isinstance(tool_use.input, dict) and (
                        tool_use.input.get(CLEARED_TOOL_USE_KEY) is True
                    )
            except Exception:
                continue
        return False

    def _is_context_edit_message(self, msg: CfMessage) -> bool:
        """Check if message is related to context_edit tool use."""
        tool_use_ids = self._extract_tool_use_ids_from_message(msg)
        return bool(tool_use_ids & set(self._context_edit_tool_uses.keys()))

    async def _delete_context_edit_messages(self, context: AnalectRunContext) -> None:
        """
        Delete all context_edit tool messages from conversation history.

        For each tool_use_id in _context_edit_tool_uses, deletes:
        - Tool use message (AI message)
        - Tool result message (HUMAN message)
        - Tool use info message (SYS message with <system_info>)

        Uses a comprehensive predicate to identify context_edit messages without
        relying on _tool_use_message_map which might not be built yet.
        """
        if not self._context_edit_tool_uses:
            return

        self._build_tool_use_message_map(context)

        # 1. First: Merge fully-ignored turns from PREVIOUS context_edit operations
        if self.merge_fully_ignored_turns_enabled:
            merge_fully_ignored_turns(
                memory_manager=context.memory_manager,
                is_ignored_fn=self._is_message_fully_processed,
            )

        # 2. Then: Ignore THIS context_edit's tool uses
        for tool_use_id, tool_use_and_result in self._context_edit_tool_uses.items():
            tool_use_msgs = self._get_tool_use_messages(tool_use_id)
            await self._ignore_tool_use_messages(
                tool_use_msgs,
                tool_use_id,
                context,
                placeholder_text=Tag(
                    name="system",
                    contents=f"A function call to `{tool_use_and_result.tool_use.name}` was cleared. "
                    + (
                        "The context window failed to optimize."
                        if tool_use_and_result.tool_result.is_error is True
                        else "The context window has been successfully optimized. Please continue the conversation."
                    ),
                ).prettify(),
            )

        # 3. Clear the tracking list
        self._context_edit_tool_uses.clear()

    async def on_before_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> None:
        # Reset continuous edit counter when using non-context_edit tool
        if tool_use.name != CONTEXT_EDIT_TOOL_NAME:
            self._continuous_edit_count = 0

        # Enforce context management if input tokens exceed threshold
        last_prompt_token_length = self.get_last_prompt_token_length() or 0
        if last_prompt_token_length >= self.input_tokens_trigger:
            if tool_use.name not in MEMORY_TOOL_NAMES + [CONTEXT_EDIT_TOOL_NAME]:
                # Set flag before raising interruption - it will be read in the next
                # on_invoke_llm_with_params call to inject the enforcement message
                self._enforce_mode = True
                raise OrchestratorInterruption()

    @override
    async def on_after_tool_use_result(
        self,
        tool_use: ant.MessageContentToolUse,
        tool_result: ant.MessageContentToolResult,
        context: AnalectRunContext,
    ) -> None:
        # Check if tool_use_info already added (deduplication via flag)
        # This extension is both ToolUseExtension and ToolUseObserver, so it will be called twice
        if self._last_added_tool_use_id == tool_result.tool_use_id:
            return  # Already added, skip

        # Get estimated token lengths for tool_use and tool_result
        tool_use_tokens, tool_result_tokens = await self.get_prompt_token_lengths(
            [
                AIMessage(content=[tool_use.dict()]),
                HumanMessage(content=[tool_result.dict()]),
            ]
        )

        # Count image blocks in the tool result and calculate their token cost
        image_count = 0
        image_tokens = 0
        if isinstance(tool_result.content, list):
            for item in tool_result.content:
                if isinstance(item, ant.MessageContentImage):
                    dims = get_image_dimensions_from_block(item.model_dump(mode="json"))
                    if dims is not None:
                        image_tokens += calculate_image_tokens_from_dimensions(dims[0], dims[1])
                    image_count += 1

        # Register this tool use (needed for context edit logging)
        registry_entry = {
            "tool_use_id": tool_result.tool_use_id,
            "tool_name": tool_use.name,
            "created_at_turn": self._turn_counter,
            "input_tokens": tool_use_tokens,
            "result_tokens": tool_result_tokens,
            "image_count": image_count,
            "image_tokens": image_tokens,
        }
        self._tool_use_registry[tool_result.tool_use_id] = registry_entry

        # Training data: write registry entry and add previews
        if self.training_data_dir is not None:
            input_preview = self._extract_content_preview(tool_use.input)
            result_preview_text = ""
            if isinstance(tool_result.content, list):
                text_parts = []
                for item in tool_result.content:
                    if isinstance(item, ant.MessageContentText):
                        text_parts.append(item.text[:200])
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", ""))[:200])
                result_preview_text = " ".join(text_parts)[:200]
            elif isinstance(tool_result.content, str):
                result_preview_text = tool_result.content[:200]

            registry_entry["input_preview"] = input_preview
            registry_entry["result_preview"] = result_preview_text
            self._log_training_data("tool_use_registry.jsonl", {
                "event": "tool_use_registered",
                **registry_entry,
            })

        # Build system_info tag contents
        sys_info_contents: list[Tag | str] = [
            Tag(
                name="tool_use_id",
                contents=tool_result.tool_use_id,
            ),
            Tag(
                name="estimated_tokens",
                contents=f"tool_use: {tool_use_tokens}, tool_result: {tool_result_tokens}, total: {tool_use_tokens + tool_result_tokens}",
            ),
        ]
        if image_count > 0:
            sys_info_contents.append(
                Tag(
                    name="images",
                    contents=f"{image_count} image{'s' if image_count != 1 else ''} (~{image_tokens} token{'s' if image_tokens != 1 else ''} of {tool_result_tokens} tool_result tokens)",
                )
            )

        # Add cumulative context usage if enabled
        if self.enable_context_usage:
            last_prompt_length = self.get_last_prompt_token_length()
            if last_prompt_length is not None:
                usage_pct = (last_prompt_length / self.context_window_size) * 100
                sys_info_contents.append(
                    Tag(
                        name="context_usage",
                        contents=f"current: {last_prompt_length:,} tokens, window: {self.context_window_size:,} tokens ({usage_pct:.1f}%)",
                    )
                )

        # Create and add sys info message
        tool_use_info = CfMessage(
            type=cf.MessageType.SYS,
            content=Tag(
                name="system_info",
                contents=sys_info_contents,
            ).prettify(),
            additional_kwargs={SYS_INFO_TOOL_USE_ID_KEY: tool_result.tool_use_id},
        )
        context.memory_manager.add_messages([tool_use_info])

        # Mark as added to prevent duplicate
        self._last_added_tool_use_id = tool_result.tool_use_id

        # Record context_edit tool use IDs for cleanup
        if tool_use.name == CONTEXT_EDIT_TOOL_NAME:
            self._context_edit_tool_uses[tool_use.id] = _ToolUseResult(
                tool_use=tool_use,
                tool_result=tool_result,
            )

        # Clean up context_edit tool messages if enabled and no pending edits
        if (
            tool_use.name == CONTEXT_EDIT_TOOL_NAME
            and self.remove_context_edit_history
            and not self._pending_data.edits
        ):
            await self._delete_context_edit_messages(context)

    def _merge_pending_with_new_edits(
        self, new_edits: list[ContextEdit]
    ) -> tuple[list[ContextEdit], int, list[tuple[str, ContextEditOperation]]]:
        """
        Merge pending edits with new edits, deduplicating by tool_use_id.

        New edits override old edits for the same tool_use_id.

        Args:
            new_edits: New edits from current context_edit call

        Returns:
            Tuple of (all_edits, pending_count, replaced_edits) where:
            - all_edits: Merged and deduplicated edits
            - pending_count: Number of edits that were pending before merge
            - replaced_edits: List of (tool_use_id, old_operation) for replaced edits
        """
        pending_count = len(self._pending_data.edits)
        if not self._pending_data.edits:
            return new_edits, 0, []

        # Track which edits are being replaced
        replaced_edits: list[tuple[str, ContextEditOperation]] = []

        # Create a dict to deduplicate by tool_use_id (new edits override old)
        edit_map = {edit.tool_use_id: edit for edit in self._pending_data.edits}
        for edit in new_edits:
            if edit.tool_use_id in edit_map:
                # This is a replacement
                old_edit = edit_map[edit.tool_use_id]
                replaced_edits.append((edit.tool_use_id, old_edit.operation))
            edit_map[edit.tool_use_id] = edit  # New overrides old
        all_edits = list(edit_map.values())
        return all_edits, pending_count, replaced_edits

    async def _calculate_replacement_info(
        self,
        replaced_edits: list[tuple[str, ContextEditOperation]],
        new_edits: list[ContextEdit],
        context: AnalectRunContext,
    ) -> list[ReplacementInfo]:
        """
        Calculate replacement info with token savings delta.

        Args:
            replaced_edits: List of (tool_use_id, old_operation) for replaced edits
            new_edits: New edits from current context_edit call
            context: Analect run context

        Returns:
            List of ReplacementInfo with token savings delta
        """
        replacement_info: list[ReplacementInfo] = []

        for tool_use_id, old_operation in replaced_edits:
            # Find the new edit
            new_edit = next(
                (e for e in new_edits if e.tool_use_id == tool_use_id), None
            )
            if not new_edit:
                continue

            # Get old savings from map
            old_savings = self._pending_data.edit_savings_map.get(tool_use_id, 0)

            # Calculate new savings
            new_savings = await self._estimate_edit_token_savings(new_edit, context)

            # Calculate delta
            delta = new_savings - old_savings

            replacement_info.append(
                ReplacementInfo(
                    tool_use_id=tool_use_id,
                    old_operation=old_operation,
                    new_operation=new_edit.operation,
                    old_savings=old_savings,
                    new_savings=new_savings,
                    delta=delta,
                )
            )

        return replacement_info

    async def _handle_empty_edits_with_pending(
        self, tool_use_id: str, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """
        Handle case where new edits are empty but pending edits exist.

        Keeps pending and returns pending message asking for more edits.

        Args:
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with pending message
        """
        msg = self._format_pending_message(
            pending_count=len(self._pending_data.edits),
            estimated_savings=self._pending_data.estimated_savings,
            rejected_edits=[],
        )
        await context.io.system(
            msg,
            run_label="Context Management",
            run_status=cf.RunStatus.IN_PROGRESS,
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=msg,
            is_error=False,
        )

    async def _handle_no_edits(
        self, tool_use_id: str, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """
        Handle case where no edits were provided at all.

        Args:
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result indicating no changes
        """
        msg = "No edits provided; no changes applied."
        await context.io.system(
            msg,
            run_label="Context Management",
            run_status=cf.RunStatus.COMPLETED,
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=msg,
            is_error=False,
        )

    async def _show_processing_message(
        self,
        all_edits: list[ContextEdit],
        pending_count: int,
        new_edits: list[ContextEdit],
        context: AnalectRunContext,
    ) -> None:
        """
        Show processing message indicating operation has started.

        Args:
            all_edits: All edits being processed (pending + new)
            pending_count: Number of edits that were pending
            new_edits: New edits from current call
            context: Analect run context
        """
        edit_msg = f"Processing {len(all_edits)} context edit(s)"
        if pending_count > 0:
            edit_msg += f" ({pending_count} pending + {len(new_edits)} new)"
        edit_msg += "..."
        await context.io.system(
            edit_msg,
            run_label="Context Management",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

    async def _handle_pending_validation(
        self,
        validation_result: ValidationResult,
        edit_savings_map: dict[str, int],
        tool_use_id: str,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Handle pending validation result.

        Stores valid edits and returns pending message asking for more.
        If max continuous edits exceeded, triggers auto-force instead.

        Args:
            validation_result: Result with is_pending=True
            edit_savings_map: Map of tool_use_id to individual savings
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with pending message or auto-force result
        """
        # Check if max continuous edits exceeded - trigger auto-force
        if self._continuous_edit_count >= self.max_continuous_edits:
            return await self._handle_auto_force_edits(
                validation_result, edit_savings_map, tool_use_id, context
            )

        # Normal pending flow
        self._pending_data.edits = validation_result.valid_edits
        self._pending_data.estimated_savings = validation_result.estimated_savings
        self._pending_data.edit_savings_map = edit_savings_map
        pending_msg = self._format_pending_message(
            pending_count=len(validation_result.valid_edits),
            estimated_savings=validation_result.estimated_savings,
            rejected_edits=validation_result.rejected_edits,
            replacement_info=validation_result.replacement_info,
        )
        self._log_context_edit({
            "event": "context_edit_pending",
            "tool_use_id": tool_use_id,
            "pending_edits": len(validation_result.valid_edits),
            "estimated_savings": validation_result.estimated_savings,
            "rejected_count": len(validation_result.rejected_edits),
        })
        await context.io.system(
            pending_msg,
            run_label="Context Management",
            run_status=cf.RunStatus.WARNING,
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=pending_msg,
            is_error=True,
        )

    async def _handle_validation_failure(
        self,
        validation_result: ValidationResult,
        tool_use_id: str,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Handle hard validation failure.

        Clears pending edits and returns error message.

        Args:
            validation_result: Result with is_safe=False
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with error message
        """
        self._pending_data.clear()  # Clear pending on hard failure
        error_msg = self._format_validation_failure(validation_result)
        self._log_context_edit({
            "event": "context_edit_validation_failure",
            "tool_use_id": tool_use_id,
            "failure_reason": validation_result.failure_reason,
            "rejected_count": len(validation_result.rejected_edits),
        })
        await context.io.system(
            error_msg,
            run_label="Context Management",
            run_status=cf.RunStatus.WARNING,
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=error_msg,
            is_error=True,
        )

    async def _handle_successful_edits(
        self,
        validation_result: ValidationResult,
        pending_count: int,
        tool_use_id: str,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Handle successful edits.

        Applies all edits, clears pending, and returns success message.

        Args:
            validation_result: Result with is_safe=True
            pending_count: Number of edits that were pending before success
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with success message
        """
        edits_to_apply = validation_result.valid_edits

        # Run compression agents if enabled
        if self.compression_agent_enabled:
            edits_to_apply = await self._run_compression_agents(
                edits_to_apply, context
            )

        stats, edit_items = await self._apply_edits(edits_to_apply, context)
        self._pending_data.clear()  # Clear pending on success
        self._continuous_edit_count = 0  # Reset counter on success

        self._log_context_edit({
            "event": "context_edit_applied",
            "tool_use_id": tool_use_id,
            "turn_number": self._turn_counter,
            "tokens_before": stats.tokens_before,
            "tokens_after": stats.tokens_after,
            "tokens_saved": stats.tokens_saved,
            "savings_percent": round(stats.savings_percent, 1),
            "rewritten_count": stats.rewritten_count,
        })

        # Write consolidated JSON record
        self._write_context_edit_json({
            "event": "context_edit_applied",
            "tool_use_id": tool_use_id,
            "turn_number": self._turn_counter,
            "pending_count": pending_count,
            "tokens_before": stats.tokens_before,
            "tokens_after": stats.tokens_after,
            "tokens_saved": stats.tokens_saved,
            "savings_percent": round(stats.savings_percent, 1),
            "rewritten_count": stats.rewritten_count,
            "edits": edit_items,
        })

        # Training data: record edit decisions with full context
        if self.training_data_dir is not None:
            self._edit_history.append({
                "turn_number": self._turn_counter,
                "tokens_before": stats.tokens_before,
                "tokens_after": stats.tokens_after,
                "tokens_saved": stats.tokens_saved,
                "edits": edit_items,
            })

        # Show success with statistics
        result = self._format_success_message(stats, validation_result, pending_count)
        await context.io.system(
            result,
            run_label="Context Management",
            run_status=cf.RunStatus.COMPLETED,
        )

        # Reset cache checkpoint after context edits to enable proper cache hits
        # The post-edit state is the new stable prefix that should be cached
        if context.runnable is not None:
            # Get BasePromptCaching extension from the orchestrator
            if isinstance(context.runnable, BaseOrchestrator):
                # Note: Only one BasePromptCaching extension should be configured per orchestrator
                # We process the first one found and break to avoid duplicate resets
                for ext in context.runnable.extensions:
                    if isinstance(ext, BasePromptCaching):
                        # Reset last_checkpoint to the new token count
                        messages = context.memory_manager.get_session_memory().messages
                        total_length = stats.tokens_after
                        messages[-1].additional_kwargs[CACHE_BREAKPOINT_KEY] = True
                        ext.set_last_checkpoint(total_length)
                        # Invalidate cached token length so on_memory() uses fresh value
                        ext.set_last_prompt_token_length(total_length)
                        break

        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=result,
            is_error=False,
        )

    def _should_include_tool_use(
        self, tool_use_id: str, tool_use_msg: CfMessage | None
    ) -> bool:
        """
        Check if a tool use should be included in context management operations.

        Excludes:
        - Tool uses with no message (None)
        - Already-ignored tool uses
        - context_edit tool uses themselves

        Args:
            tool_use_id: The tool use ID
            tool_use_msg: The tool use message (can be None)

        Returns:
            True if tool use should be included, False otherwise
        """
        if tool_use_msg is None:
            return False

        # Skip if already ignored
        if tool_use_id in self._ignored_tool_use_ids:
            return False

        # Skip context_edit tool uses
        tool_name = self._extract_tool_name(tool_use_id, tool_use_msg)
        if tool_name == CONTEXT_EDIT_TOOL_NAME:
            return False

        return True

    def _get_oldest_non_ignored_tool_uses(
        self,
        context: AnalectRunContext,
        exclude_ids: set[str],
    ) -> list[str]:
        """
        Get tool_use_ids sorted by message order (oldest first).

        Excludes:
        - Already-ignored tool uses
        - Tool uses in exclude_ids (already pending)
        - context_edit tool uses themselves

        Args:
            context: The analect run context
            exclude_ids: Set of tool_use_ids to exclude (already in pending)

        Returns:
            List of tool_use_ids from oldest to newest
        """
        messages = context.memory_manager.get_session_memory().messages
        ordered_ids: list[str] = []

        for msg in messages:
            if msg.type != cf.MessageType.AI:
                continue

            tool_use_ids = self._extract_tool_use_ids_from_message(msg)
            for tool_use_id in tool_use_ids:
                # Skip if in exclude set
                if tool_use_id in exclude_ids:
                    continue

                # Get tool use messages
                tool_use_msgs = self._get_tool_use_messages(tool_use_id)
                tool_use_msg = tool_use_msgs.tool_use

                # Use shared filtering logic
                if not self._should_include_tool_use(tool_use_id, tool_use_msg):
                    continue

                ordered_ids.append(tool_use_id)

        return ordered_ids

    async def _auto_force_rewrite_edits(
        self,
        context: AnalectRunContext,
        current_savings: int,
        valid_edits: list[ContextEdit],
    ) -> tuple[list[ContextEdit], int]:
        """
        Automatically generate REWRITE edits for oldest tool uses until threshold is met.

        Auto-generated rewrites use minimal descriptions based on tool name and input.

        Args:
            context: The analect run context
            current_savings: Current estimated token savings from pending edits
            valid_edits: Current cycle's validated edits to exclude from auto-force

        Returns:
            Tuple of (forced_edits, additional_savings)
        """
        remaining_needed = self.practical_clear_at_least - current_savings
        if remaining_needed <= 0:
            return [], 0

        pending_ids = {edit.tool_use_id for edit in valid_edits}
        oldest_ids = self._get_oldest_non_ignored_tool_uses(context, pending_ids)

        forced_edits: list[ContextEdit] = []
        additional_savings = 0

        for tool_use_id in oldest_ids:
            if additional_savings >= remaining_needed:
                break

            tool_use_msgs = self._get_tool_use_messages(tool_use_id)
            tool_use_msg = tool_use_msgs.tool_use
            if tool_use_msg is None:
                continue

            # Generate minimal replacement based on tool name
            tool_name = self._extract_tool_name(tool_use_id, tool_use_msg)
            original_input = self._extract_tool_use_input(tool_use_msg, tool_use_id)
            replacement = self._generate_auto_rewrite(tool_name, original_input)

            edit = ContextEdit(
                tool_use_id=tool_use_id,
                operation=ContextEditOperation.REWRITE,
                guidance="Totally irrelevant — compress to a one-liner",
                tool_result_content_replacement=replacement,
                reason="Auto-forced due to max continuous edits exceeded",
            )

            savings = await self._estimate_edit_token_savings(edit, context)
            forced_edits.append(edit)
            additional_savings += savings

        return forced_edits, additional_savings

    def _generate_auto_rewrite(self, tool_name: str, original_input: dict[str, Any] | None) -> str:
        """Generate a minimal rewrite replacement for auto-forced edits."""
        if original_input is None:
            return f"[Auto-rewritten] Ran {tool_name}. Output cleared to save context."

        if tool_name == "bash":
            command = str(original_input.get("command", ""))[:150]
            return f"[Auto-rewritten] Ran bash: {command}. Output cleared to save context."
        elif tool_name == "str_replace_based_edit_tool":
            file_path = original_input.get("path", original_input.get("file_path", ""))
            cmd = original_input.get("command", "view")
            return f"[Auto-rewritten] {cmd} {file_path}. Content cleared to save context."
        else:
            return f"[Auto-rewritten] Ran {tool_name}. Output cleared to save context."

    async def _show_auto_force_notification(
        self,
        forced_edits: list[ContextEdit],
        total_savings: int,
        pending_count: int,
        context: AnalectRunContext,
    ) -> None:
        """
        Show UI notification about auto-force triggering.

        Args:
            forced_edits: List of auto-forced edits
            total_savings: Total estimated token savings
            pending_count: Number of pending edits before auto-force
            context: The analect run context
        """
        forced_list = "\n".join(f"- {edit.tool_use_id}" for edit in forced_edits)
        message = AUTO_FORCE_MESSAGE_TEMPLATE.format(
            max_attempts=self.max_continuous_edits,
            forced_count=len(forced_edits),
            forced_list=forced_list,
            pending_count=pending_count,
            total_savings=total_savings,
        )
        await context.io.system(
            message,
            run_label="Context Management",
            run_status=cf.RunStatus.WARNING,
        )

    def _format_auto_force_success_message(
        self,
        stats: EditStats,
        forced_edits: list[ContextEdit],
    ) -> str:
        """
        Format success message noting which tool uses were auto-forced.

        Args:
            stats: Statistics about the applied edits
            forced_edits: List of auto-forced edits

        Returns:
            Formatted success message with auto-force details
        """
        auto_rewritten_list = ", ".join(edit.tool_use_id for edit in forced_edits)

        return f"""✅ **Context Edit Applied** (with auto-forced rewrites)

**Results:**
- Rewritten: {stats.rewritten_count} tool use(s)
- Kept unchanged: {stats.kept_count} tool use(s)

**Token savings:**
- Before: ~{stats.tokens_before or 0:,} tokens
- After: ~{stats.tokens_after or 0:,} tokens
- Saved: ~{stats.tokens_saved or 0:,} tokens ({stats.savings_percent or 0.0:.1f}%)

**Auto-rewritten tool uses ({len(forced_edits)}):** {auto_rewritten_list}
"""

    async def _handle_auto_force_edits(
        self,
        validation_result: ValidationResult,
        edit_savings_map: dict[str, int],
        tool_use_id: str,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Handle auto-forcing REWRITE edits when max attempts exceeded.

        Args:
            validation_result: Current validation result with pending edits
            edit_savings_map: Map of tool_use_id to individual savings
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with success or partial success message
        """
        # Generate forced edits
        forced_edits, additional_savings = await self._auto_force_rewrite_edits(
            context, validation_result.estimated_savings, validation_result.valid_edits
        )

        pending_count = len(validation_result.valid_edits)

        # If no more tool uses to force-rewrite, apply what we have
        if not forced_edits:
            # Apply pending edits even if below threshold - exit gracefully
            edits_to_apply = validation_result.valid_edits
            if self.compression_agent_enabled:
                edits_to_apply = await self._run_compression_agents(edits_to_apply, context)
            stats, edit_items = await self._apply_edits(edits_to_apply, context)
            self._pending_data.clear()
            self._continuous_edit_count = 0

            self._write_context_edit_json({
                "event": "context_edit_applied",
                "tool_use_id": tool_use_id,
                "turn_number": self._turn_counter,
                "auto_forced": True,
                "partial": True,
                "tokens_before": stats.tokens_before,
                "tokens_after": stats.tokens_after,
                "tokens_saved": stats.tokens_saved,
                "savings_percent": round(stats.savings_percent, 1),
                "rewritten_count": stats.rewritten_count,
                "edits": edit_items,
            })

            result = f"""⚠️ **Context Edit Applied** (partial - threshold not met)

No more tool uses available to auto-rewrite.
Applied {len(validation_result.valid_edits)} pending edit(s).

**Results:**
- Rewritten: {stats.rewritten_count} tool use(s)
- Kept unchanged: {stats.kept_count} tool use(s)

**Token savings:**
- Before: ~{stats.tokens_before or 0:,} tokens
- After: ~{stats.tokens_after or 0:,} tokens
- Saved: ~{stats.tokens_saved or 0:,} tokens ({stats.savings_percent or 0.0:.1f}%)
"""
            await context.io.system(
                result,
                run_label="Context Management",
                run_status=cf.RunStatus.COMPLETED,
            )
            return ant.MessageContentToolResult(
                tool_use_id=tool_use_id,
                content=result,
                is_error=False,
            )

        # Combine pending + forced edits
        all_edits = validation_result.valid_edits + forced_edits
        total_savings = validation_result.estimated_savings + additional_savings

        # Show auto-force notification
        await self._show_auto_force_notification(
            forced_edits, total_savings, pending_count, context
        )

        # Run compression agents on pending edits (forced edits already have replacements)
        if self.compression_agent_enabled:
            compressed_pending = await self._run_compression_agents(
                validation_result.valid_edits, context
            )
            all_edits = compressed_pending + forced_edits

        # Apply all edits
        stats, edit_items = await self._apply_edits(all_edits, context)
        self._pending_data.clear()
        self._continuous_edit_count = 0

        self._write_context_edit_json({
            "event": "context_edit_applied",
            "tool_use_id": tool_use_id,
            "turn_number": self._turn_counter,
            "auto_forced": True,
            "forced_count": len(forced_edits),
            "tokens_before": stats.tokens_before,
            "tokens_after": stats.tokens_after,
            "tokens_saved": stats.tokens_saved,
            "savings_percent": round(stats.savings_percent, 1),
            "rewritten_count": stats.rewritten_count,
            "edits": edit_items,
        })

        # Format success message with auto-force note
        result = self._format_auto_force_success_message(stats, forced_edits)
        await context.io.system(
            result,
            run_label="Context Management",
            run_status=cf.RunStatus.COMPLETED,
        )

        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=result,
            is_error=False,
        )

    @override
    async def on_tool_use(
        self,
        tool_use: ant.MessageContentToolUse,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Execute context_edit tool.

        Directly manipulates context.memory_manager.memory.messages

        Flow:
        1. Parse input and merge with pending edits
        2. Handle special cases (empty edits, no edits)
        3. Validate all edits
        4. Handle validation outcome (pending, failure, or success)
        """
        # Reset enforce mode flag
        self._enforce_mode = False

        # Increment continuous edit counter
        self._continuous_edit_count += 1

        # Parse and merge edits
        input_data = ContextEditInput.model_validate(tool_use.input)
        new_edits = input_data.edits
        all_edits, pending_count, replaced_edits = self._merge_pending_with_new_edits(
            new_edits
        )

        self._log_context_edit({
            "event": "context_edit_requested",
            "tool_use_id": tool_use.id,
            "new_edits": [e.model_dump(mode="json", exclude_none=True) for e in new_edits],
            "total_edits": len(all_edits),
            "pending_count": pending_count,
        })

        # Handle empty edits with pending
        if not new_edits and self._pending_data.edits:
            return await self._handle_empty_edits_with_pending(tool_use.id, context)

        # Handle no edits at all
        if not all_edits:
            return await self._handle_no_edits(tool_use.id, context)

        # Show progress and validate
        await self._show_processing_message(
            all_edits, pending_count, new_edits, context
        )
        self._build_tool_use_message_map(context)

        # Calculate replacement info before validation
        replacement_info = await self._calculate_replacement_info(
            replaced_edits, new_edits, context
        )

        validation_result = await self._validate_edits(
            all_edits, context, replacement_info
        )

        # Handle different validation outcomes
        if validation_result.is_pending:
            return await self._handle_pending_validation(
                validation_result,
                validation_result.edit_savings_map,
                tool_use.id,
                context,
            )
        if not validation_result.is_safe:
            return await self._handle_validation_failure(
                validation_result, tool_use.id, context
            )

        return await self._handle_successful_edits(
            validation_result, pending_count, tool_use.id, context
        )

    @override
    async def on_process_messages_complete(self, context: AnalectRunContext) -> None:
        """Check if progress needs to be updated and raise interruption if not complete"""
        if not self._pending_data.edits or (
            (self.get_last_prompt_token_length() or 0) < self.input_tokens_trigger
        ):
            return

        reminder_message = self._format_pending_reminder_message(
            pending_count=len(self._pending_data.edits),
            estimated_savings=self._pending_data.estimated_savings,
        )
        raise OrchestratorInterruption(reminder_message)

    def _format_success_message(
        self,
        stats: EditStats,
        validation_result: ValidationResult,
        pending_count: int = 0,
    ) -> str:
        """
        Format success message with statistics and warnings.

        Override this method to customize the success message format.

        Args:
            stats: Statistics about the applied edits
            validation_result: Validation result containing rejected edits
            pending_count: Number of edits that were pending before this success

        Returns:
            Formatted success message string
        """
        result = "✅ Context optimized"
        if pending_count > 0:
            result += f" (accumulated {pending_count} pending edits)"
        result += "\n"

        result += f"""
Changes:
- {stats.rewritten_count} tool uses rewritten
- {stats.kept_count} tool uses kept unchanged

Token Savings:
- Before: ~{stats.tokens_before or 0:,} tokens
- After: ~{stats.tokens_after or 0:,} tokens
- Saved: ~{stats.tokens_saved or 0:,} tokens ({stats.savings_percent or 0.0:.1f}% reduction)

NOTE:
* Token counts are rough estimates and exclude the tool definitions in the system prompt.
"""

        # Add warning if tokens saved is less than clear_at_least threshold
        if stats.tokens_saved < self.practical_clear_at_least:
            result += f"""

⚠️ **Warning: Insufficient token savings**

You cleared ~{stats.tokens_saved or 0:,} tokens, which is less than the recommended {self.practical_clear_at_least:,} tokens.

This may not justify the cost of invalidating the prompt cache. Next time, please try to:
- Rewrite more aggressively (shorter summaries, more omission)
- Target older, less relevant tool uses
- Rewrite large file reads and verbose command outputs

This helps make each context edit more worthwhile.
"""

        # Add rejected edits information if any
        result += self._format_rejected_edits_section(validation_result.rejected_edits)

        return result

    def _format_rejected_edits_section(self, rejected_edits: list[RejectedEdit]) -> str:
        """
        Format rejected edits section for display in messages.

        Args:
            rejected_edits: List of edits that were rejected with reasons

        Returns:
            Formatted string showing rejected edits with reasons, or empty string if none
        """
        if not rejected_edits:
            return ""

        result = f"\n\n⚠️ {len(rejected_edits)} edit(s) were rejected:\n"
        result += "\n".join(
            f"- {r.edit.tool_use_id} ({r.edit.operation.value}): {r.reason}"
            for r in rejected_edits
        )
        return result

    def _format_replacement_info_section(
        self, replacement_info: list[ReplacementInfo]
    ) -> str:
        """
        Format replacement info section showing edits that replaced pending edits.

        Args:
            replacement_info: List of replacement information

        Returns:
            Formatted string showing replacements with token delta, or empty string if none
        """
        if not replacement_info:
            return ""

        result = (
            f"\n\n⚠️ {len(replacement_info)} edit(s) replaced already-pending edits:\n"
        )
        for info in replacement_info:
            op_change = f"{info.old_operation.value} → {info.new_operation.value}"
            if info.delta > 0:
                delta_str = f"+{info.delta:,} tokens (savings increased)"
            elif info.delta < 0:
                delta_str = f"{info.delta:,} tokens (savings decreased)"
            else:
                delta_str = "±0 tokens (no change)"
            result += f"- {info.tool_use_id} ({op_change}): {delta_str}\n"

        result += "\n**Note:** Replacing edits recalculates token savings. Consider if these changes are beneficial."
        return result

    def _format_pending_message(
        self,
        pending_count: int,
        estimated_savings: int,
        rejected_edits: list[RejectedEdit],
        replacement_info: list[ReplacementInfo] | None = None,
    ) -> str:
        """
        Format pending message when edits are accumulated but savings insufficient.

        Args:
            pending_count: Number of edits currently pending
            estimated_savings: Estimated token savings from pending edits
            rejected_edits: List of edits that were rejected with reasons
            replacement_info: Optional list of replacement information

        Returns:
            Formatted pending message string
        """
        estimated_savings = estimated_savings or 0
        result = f"""⚠️ INSUFFICIENT EDITS - MORE AGGRESSIVE COMPRESSION REQUIRED

Current Status:
- {pending_count} edit(s) pending (NOT ENOUGH)
- ~{estimated_savings:,} tokens saved so far (INSUFFICIENT)
- Required minimum: {self.practical_clear_at_least:,} tokens
- Additional needed: ~{max(0, self.practical_clear_at_least - estimated_savings):,} tokens

**CRITICAL: You MUST be MORE AGGRESSIVE in context compression to avoid repeated editing loops.**

Your current rewrites are too conservative. Be significantly more aggressive:

1. **Rewrite more aggressively** - Use shorter summaries and more omission:
   - Large file reads → one-line description of what was in the file
   - Verbose command output → extract only the key result
   - Old exploratory reads → minimal annotation

2. **Prioritize OLDER content** - Recent work is more likely relevant:
   - Start from the earliest tool uses
   - Work your way forward chronologically

3. **Think in BATCHES** - Don't edit one at a time:
   - Review 5-10 tool uses at once
   - Submit comprehensive edits together

Your {pending_count} pending edit(s) are saved. Call `context_edit` NOW with significantly more aggressive edits.
"""

        # Add replacement info if any
        if replacement_info:
            result += self._format_replacement_info_section(replacement_info)

        # Add rejected edits information if any
        result += self._format_rejected_edits_section(rejected_edits)

        return result

    def _format_pending_reminder_message(
        self,
        pending_count: int,
        estimated_savings: int,
    ) -> str:
        """
        Format reminder message for interruption when pending edits exist.

        This message is shown when the orchestrator is about to exit but there
        are still pending edits that haven't reached the minimum threshold.

        Args:
            pending_count: Number of edits currently pending
            estimated_savings: Estimated token savings from pending edits
            rejected_edits: List of edits that were rejected with reasons

        Returns:
            Formatted reminder message string for interruption context
        """
        estimated_savings = estimated_savings or 0
        result = f"""⚠️ **Pending context edits incomplete**

You have {pending_count} pending context edit(s) that need to be completed.

Status:
- {pending_count} edit(s) pending
- ~{estimated_savings:,} tokens saved so far
- Target: {self.practical_clear_at_least:,} tokens minimum
- Need: ~{max(0, self.practical_clear_at_least - estimated_savings):,} more tokens

**Required Action:**
You MUST call `context_edit` again with more edits to reach the minimum threshold before you can exit.

Tips to reach the threshold:
- Rewrite more aggressively (shorter annotations, more omission)
- Target large file reads and verbose command outputs
- Focus on older, less relevant content
- Your previous {pending_count} edit(s) are already saved and will accumulate
"""

        return result

    def _format_validation_failure(self, validation_result: ValidationResult) -> str:
        """
        Format validation failure message.

        Override this method to customize the error message format.

        Args:
            validation_result: Validation result with failure_reason

        Returns:
            Formatted error message string
        """
        return f"Context edit rejected:\n{validation_result.failure_reason}"

    def _validate_tool_use_ids_exist(
        self, edits: list[ContextEdit], context: AnalectRunContext
    ) -> tuple[list[ContextEdit], list[RejectedEdit]]:
        """
        Check 1: Filter edits to only valid tool_use_ids.

        Args:
            edits: List of context edits to validate
            context: The analect run context

        Returns:
            (valid_edits, rejected_edits): Valid edits and rejected edits with reasons
        """
        valid_edits = []
        rejected_edits = []

        for edit in edits:
            tool_use_id = edit.tool_use_id
            tool_use_msgs = self._get_tool_use_messages(tool_use_id)

            if not tool_use_msgs.has_tool_use():
                rejected_edits.append(
                    RejectedEdit(edit=edit, reason="Unknown tool_use_id")
                )
            elif not tool_use_msgs.has_tool_result():
                rejected_edits.append(
                    RejectedEdit(edit=edit, reason="Tool result not found")
                )
            else:
                valid_edits.append(edit)

        return valid_edits, rejected_edits

    def _extract_tool_name(self, tool_use_id: str, tool_use_msg: CfMessage) -> str:
        """Extract tool name from tool use message."""
        tool_name = "unknown"
        if isinstance(tool_use_msg.content, list):
            for item in tool_use_msg.content:
                try:
                    tool_use_obj = ant.MessageContentToolUse.model_validate(item)
                    if tool_use_obj.id == tool_use_id:
                        tool_name = tool_use_obj.name
                        break
                except Exception:
                    continue
        return tool_name

    def _extract_tool_use_input(self, tool_use_msg: CfMessage, tool_use_id: str) -> dict[str, Any] | None:
        """Extract tool use input dict from a tool_use message."""
        if isinstance(tool_use_msg.content, list):
            for item in tool_use_msg.content:
                try:
                    tool_use_obj = ant.MessageContentToolUse.model_validate(item)
                    if tool_use_obj.id == tool_use_id:
                        if isinstance(tool_use_obj.input, dict):
                            return tool_use_obj.input
                except Exception:
                    continue
        return None

    async def _estimate_edit_token_savings(
        self, edit: ContextEdit, context: AnalectRunContext
    ) -> int:
        """Calculate token savings for a single REWRITE edit."""
        tool_use_id = edit.tool_use_id
        tool_use_msgs = self._get_tool_use_messages(tool_use_id)
        savings = 0

        # Tool result savings
        tool_result = tool_use_msgs.tool_result
        if tool_result is not None and edit.tool_result_content_replacement is not None:
            original_size = (await self.get_prompt_token_lengths([tool_result]))[0]
            temp_msg = CfMessage(
                type=cf.MessageType.HUMAN,
                content=[
                    ant.MessageContentToolResult(
                        tool_use_id=tool_use_id,
                        content=edit.tool_result_content_replacement,
                    ).model_dump(mode="json", exclude_none=True)
                ],
            )
            new_size = (await self.get_prompt_token_lengths([temp_msg]))[0]
            savings += max(0, original_size - new_size)

        # Tool use input savings (if replacement provided)
        tool_use = tool_use_msgs.tool_use
        if edit.tool_use_input_replacement is not None and tool_use is not None:
            original_size = (await self.get_prompt_token_lengths([tool_use]))[0]
            tool_name = self._extract_tool_name(tool_use_id, tool_use)
            temp_msg = CfMessage(
                type=tool_use.type,
                content=[
                    ant.MessageContentToolUse(
                        id=tool_use_id,
                        name=tool_name,
                        input={REWRITTEN_TOOL_USE_KEY: True, **edit.tool_use_input_replacement},
                    ).model_dump(mode="json", exclude_none=True)
                ],
            )
            new_size = (await self.get_prompt_token_lengths([temp_msg]))[0]
            savings += max(0, original_size - new_size)

        # Tool use info is always deleted during rewrite
        tool_use_info = tool_use_msgs.tool_use_info
        if tool_use_info is not None:
            savings += (await self.get_prompt_token_lengths([tool_use_info]))[0]

        return savings

    async def _estimate_context_edit_cleanup_savings(
        self, context: AnalectRunContext
    ) -> int:
        """
        Estimate token savings from cleaning up tracked context_edit tool calls.

        When remove_context_edit_history=True, successful context edits trigger
        cleanup of all tracked context_edit tool call messages. This method
        calculates the token savings from that cleanup.

        Note: Assumes _tool_use_message_map is already built (it is built in
        on_tool_use before validation, line 964).

        Args:
            context: The analect run context

        Returns:
            Estimated token savings from removing all tracked context_edit messages,
            or 0 if remove_context_edit_history is False or no tracked calls exist.
        """
        if not self.remove_context_edit_history or not self._context_edit_tool_uses:
            return 0

        total_savings = 0
        # Map is already built at line 964 before validation, no need to rebuild

        for tool_use_id in self._context_edit_tool_uses.keys():
            tool_use_msgs = self._get_tool_use_messages(tool_use_id)

            # Collect messages that will be deleted
            messages_to_count = []
            if tool_use_msgs.tool_use is not None:
                messages_to_count.append(tool_use_msgs.tool_use)
            if tool_use_msgs.tool_result is not None:
                messages_to_count.append(tool_use_msgs.tool_result)
            if tool_use_msgs.tool_use_info is not None:
                messages_to_count.append(tool_use_msgs.tool_use_info)

            if messages_to_count:
                token_lengths = await self.get_prompt_token_lengths(messages_to_count)
                total_savings += sum(token_lengths)

        return total_savings

    async def _validate_minimum_savings(
        self, edits: list[ContextEdit], context: AnalectRunContext
    ) -> tuple[int, bool, dict[str, int]]:
        """Check 3: Validate estimated token savings meets threshold.

        Returns:
            Tuple of (estimated_savings, should_pend, edit_savings_map):
            - estimated_savings: Estimated token savings from the edits plus cleanup
            - should_pend: Whether to pend edits (True) or not (False)
            - edit_savings_map: Map of tool_use_id to individual savings
        """
        # Parallel token estimation for better performance
        savings_list = await asyncio.gather(
            *[self._estimate_edit_token_savings(edit, context) for edit in edits]
        )

        # Create map of individual savings
        edit_savings_map = {
            edit.tool_use_id: savings for edit, savings in zip(edits, savings_list)
        }
        estimated_savings = sum(savings_list)

        # Add savings from cleaning up context_edit tool calls
        # When edits are applied successfully, all tracked context_edit messages
        # will be removed, so we include those savings in the total
        cleanup_savings = await self._estimate_context_edit_cleanup_savings(context)
        estimated_savings += cleanup_savings

        # If not enforcing, pass validation
        if not self.enforce_clear_at_least and (
            (self.get_last_prompt_token_length() or 0) >= self.input_tokens_trigger
        ):
            return (estimated_savings, False, edit_savings_map)

        # If savings insufficient, return pending state
        if estimated_savings < self.practical_clear_at_least:
            return (estimated_savings, True, edit_savings_map)

        # Savings sufficient
        return (estimated_savings, False, edit_savings_map)

    def _filter_valid_replacement_info(
        self,
        replacement_info: list[ReplacementInfo] | None,
        valid_edits: list[ContextEdit],
    ) -> list[ReplacementInfo]:
        """
        Filter replacement info to only include valid edits.

        Filters out replacements where the new edit was rejected (tool_use_id not in valid_edits).

        Args:
            replacement_info: Raw replacement info (may include rejected edits)
            valid_edits: List of edits that passed validation

        Returns:
            Filtered list of ReplacementInfo
        """
        if not replacement_info:
            return []

        valid_tool_use_ids = {edit.tool_use_id for edit in valid_edits}

        return [
            info
            for info in replacement_info
            if info.tool_use_id in valid_tool_use_ids
        ]

    async def _validate_edits(
        self,
        edits: list[ContextEdit],
        context: AnalectRunContext,
        replacement_info: list[ReplacementInfo] | None = None,
    ) -> ValidationResult:
        """
        Validate context edits before applying.

        Safety checks:
        1. Filter to valid tool_use_ids (forgiving - rejects invalid, keeps valid)
        2. Minimum tool uses retention (strict - rejects all if violated)
        3. Estimated token savings meets clear_at_least threshold (may pend or fail)

        Returns:
            ValidationResult with valid_edits and rejected_reasons
        """
        # Check 1: Filter to valid tool_use_ids (forgiving)
        valid_edits, rejected_edits = self._validate_tool_use_ids_exist(edits, context)

        # If no valid edits remain after filtering, reject all
        if not valid_edits:
            rejected_summary = "\n".join(
                f"- {r.edit.tool_use_id} ({r.edit.operation.value}): {r.reason}"
                for r in rejected_edits
            )
            return ValidationResult(
                is_safe=False,
                failure_reason=f"All edits were rejected due to invalid tool_use_ids:\n{rejected_summary}",
                valid_edits=[],
                rejected_edits=rejected_edits,
            )

        # Check 2: Minimum savings (may pend)
        # Skip savings validation when compression agent is enabled,
        # since tool_result_content_replacement is not yet available
        if self.compression_agent_enabled:
            estimated_savings = 0
            should_pend = False
            edit_savings_map = {}
        else:
            (
                estimated_savings,
                should_pend,
                edit_savings_map,
            ) = await self._validate_minimum_savings(valid_edits, context)

        # If savings fails but should pend, return pending state
        if should_pend:
            # Filter replacement info to only valid edits
            filtered_replacement_info = self._filter_valid_replacement_info(
                replacement_info, valid_edits
            )
            return ValidationResult(
                is_safe=False,
                is_pending=True,
                valid_edits=valid_edits,
                rejected_edits=rejected_edits,
                edit_savings_map=edit_savings_map,
                replacement_info=filtered_replacement_info,
                pending_reason=(
                    f"Estimated token savings ({estimated_savings:,}) is less than required "
                    f"minimum ({self.practical_clear_at_least:,}). Please add more edits to reach the threshold."
                ),
                estimated_savings=estimated_savings,
            )

        # All validation passed
        # Filter replacement info to only valid edits
        filtered_replacement_info = self._filter_valid_replacement_info(
            replacement_info, valid_edits
        )
        return ValidationResult(
            is_safe=True,
            valid_edits=valid_edits,
            rejected_edits=rejected_edits,
            failure_reason=None,
            estimated_savings=estimated_savings,
            edit_savings_map=edit_savings_map,
            replacement_info=filtered_replacement_info,
        )

    def _extract_full_tool_result_content(
        self, tool_use_id: str, tool_result_msg: CfMessage | None
    ) -> str:
        """Extract full tool result content as a string for the compression agent."""
        if tool_result_msg is None or not isinstance(tool_result_msg.content, list):
            return ""
        for item in tool_result_msg.content:
            try:
                tr = ant.MessageContentToolResult.model_validate(item)
                if tr.tool_use_id == tool_use_id:
                    return self._extract_content_preview(tr.content)
            except Exception:
                continue
        return ""

    @staticmethod
    def _add_line_numbers(content: str) -> str:
        """Add 1-based line numbers to content for the compression agent.

        Args:
            content: Raw content string.

        Returns:
            Content with each line prefixed by its line number.
        """
        lines = content.splitlines()
        width = len(str(len(lines)))
        return "\n".join(
            f"{i + 1:>{width}}: {line}" for i, line in enumerate(lines)
        )

    async def _run_compression_agent(
        self,
        original_content: str,
        guidance: str,
        token_count: int,
        context: AnalectRunContext,
    ) -> str:
        """Run the compression agent on a single tool result.

        The agent outputs structured edit instructions (DELETE/REPLACE/SUMMARY)
        instead of a full rewrite. This keeps the agent's output token count
        proportional to the amount of *change*, not the amount of retained
        content, avoiding the short-output bias that causes over-compression.

        Args:
            original_content: The original tool result content to compress
            guidance: The main agent's guidance on what to keep/omit
            token_count: Estimated token count of the original content
            context: Analect run context for LLM access

        Returns:
            Compressed content string (after applying edit instructions)
        """
        from ....core.llm_manager import LLMParams

        # Send line-numbered content so the agent can reference exact ranges
        numbered_content = self._add_line_numbers(original_content)

        user_message = COMPRESSION_AGENT_USER_PROMPT_TEMPLATE.format(
            guidance=guidance,
            content=numbered_content,
            token_count=token_count,
        )

        messages = [
            SystemMessage(content=COMPRESSION_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Use specified model, or inherit the model from the main agent
        model = self.compression_agent_model or self._current_model
        params = LLMParams(
            model=model,
            max_tokens=self.compression_agent_max_tokens,
            temperature=0.0,
            top_p=None,
        )
        chat = context.llm_manager._get_chat(params=params)
        result = await chat.ainvoke(messages)

        # Capture compression agent token usage
        try:
            metadata = result.response_metadata or {}
            usage = metadata.get("usage", {})
            self._compression_agent_usage.append({
                "model": model,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            })
        except Exception:
            pass

        raw_output = result.content if isinstance(result.content, str) else str(result.content)

        # Parse edit instructions and apply them to the original content
        parse_result = parse_edit_instructions(raw_output)

        if parse_result.errors:
            context_edit_logger.warning(
                "Compression agent parse errors: %s", parse_result.errors
            )

        if not parse_result.ops:
            # No valid instructions parsed — fall back to raw output
            # (handles edge cases where the agent outputs plain text)
            context_edit_logger.warning(
                "No edit instructions parsed from compression agent output, "
                "falling back to raw output"
            )
            return raw_output

        try:
            return apply_edit_instructions(original_content, parse_result.ops)
        except ValueError as e:
            context_edit_logger.warning(
                "Failed to apply edit instructions: %s — falling back to raw output",
                e,
            )
            return raw_output

    async def _run_compression_agents(
        self,
        edits: list[ContextEdit],
        context: AnalectRunContext,
    ) -> list[ContextEdit]:
        """Run compression agents in parallel for all edits that need compression.

        Args:
            edits: List of edits with guidance from the main agent
            context: Analect run context

        Returns:
            List of edits with tool_result_content_replacement filled in
        """
        async def compress_single(edit: ContextEdit) -> ContextEdit:
            tool_use_msgs = self._get_tool_use_messages(edit.tool_use_id)
            original_content = self._extract_full_tool_result_content(
                edit.tool_use_id, tool_use_msgs.tool_result
            )

            if not original_content:
                edit.tool_result_content_replacement = (
                    f"[empty tool result for {edit.tool_use_id}]"
                )
                return edit

            # Get token count from registry
            registry_entry = self._tool_use_registry.get(edit.tool_use_id, {})
            token_count = registry_entry.get("result_tokens", 0)

            try:
                compressed = await self._run_compression_agent(
                    original_content=original_content,
                    guidance=edit.guidance,
                    token_count=token_count,
                    context=context,
                )
                edit.tool_result_content_replacement = compressed
            except Exception as e:
                context_edit_logger.warning(
                    "Compression agent failed for %s: %s",
                    edit.tool_use_id,
                    e,
                )
                # Fall back to a basic annotation
                tool_name = registry_entry.get("tool_name", "unknown")
                edit.tool_result_content_replacement = (
                    f"[compression failed for {tool_name} tool use — {edit.guidance}]"
                )
            return edit

        # Run all compression agents in parallel
        compressed_edits = await asyncio.gather(
            *[compress_single(edit) for edit in edits]
        )
        return list(compressed_edits)

    async def _apply_edits(
        self,
        edits: list[ContextEdit],
        context: AnalectRunContext,
    ) -> tuple[EditStats, list[dict[str, Any]]]:
        """
        Apply context edits to memory.

        Directly modifies context.memory_manager.memory.messages

        Returns:
            Tuple of (EditStats, edit_items) where edit_items contains
            per-edit details for consolidated logging.
        """
        messages = context.memory_manager.get_session_memory().messages

        # Calculate tokens before using inherited token estimation
        token_lengths = await self.get_prompt_token_lengths(messages)
        tokens_before = sum(token_lengths)

        # Calculate statistics BEFORE applying edits
        total_tool_uses = sum(
            1
            for tool_use_id, msgs in self._tool_use_message_map.items()
            if msgs.tool_use is not None and tool_use_id not in self._ignored_tool_use_ids
        )
        kept_count = total_tool_uses - len(edits)

        # Apply edits
        edit_items: list[dict[str, Any]] = []
        for edit in edits:
            tool_use_id = edit.tool_use_id
            tool_use_msgs = self._get_tool_use_messages(tool_use_id)

            # Extract original content previews BEFORE applying the edit
            original_input_preview, original_result_preview = (
                self._extract_tool_use_input_and_result(
                    tool_use_id,
                    tool_use_msgs.tool_use,
                    tool_use_msgs.tool_result,
                )
            )

            # Apply REWRITE
            new_input_preview = (
                self._extract_content_preview(edit.tool_use_input_replacement)
                if edit.tool_use_input_replacement is not None
                else "unchanged"
            )
            new_result_preview = (
                self._extract_content_preview(edit.tool_result_content_replacement)
                if edit.tool_result_content_replacement is not None
                else "unchanged"
            )
            await self._rewrite_tool_use_messages(tool_use_msgs, edit, context)

            # Build per-edit item for consolidated logging
            registry_entry = self._tool_use_registry.get(tool_use_id, {})
            tool_name = registry_entry.get("tool_name") or (
                self._extract_tool_name(tool_use_id, tool_use_msgs.tool_use)
                if tool_use_msgs.tool_use is not None
                else "unknown"
            )
            item: dict[str, Any] = {
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "operation": edit.operation.value,
                "guidance": edit.guidance,
                "reason": edit.reason,
                "original_tool_use_input": original_input_preview,
                "original_tool_result_content": original_result_preview,
                "new_tool_use_input": new_input_preview,
                "new_tool_result_content": new_result_preview,
            }
            if registry_entry:
                item["age_turns"] = self._turn_counter - registry_entry.get("created_at_turn", self._turn_counter)
                item["input_tokens"] = registry_entry.get("input_tokens", 0)
                item["result_tokens"] = registry_entry.get("result_tokens", 0)
            edit_items.append(item)
            self._log_context_edit({"event": "context_edit_item_applied", **item})

        # Calculate tokens after using inherited token estimation
        messages_after = context.memory_manager.get_session_memory().messages
        token_lengths_after = await self.get_prompt_token_lengths(messages_after)
        tokens_after = sum(token_lengths_after)

        return EditStats(
            rewritten_count=len(edits),
            kept_count=kept_count,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            savings_percent=(
                ((tokens_before - tokens_after) / tokens_before * 100)
                if tokens_before > 0
                else 0
            ),
        ), edit_items

    def _compact_tool_use_input(
        self,
        message: CfMessage,
        tool_use_id: str,
        new_input: dict[str, Any],
    ) -> None:
        """Replace tool use input with compacted version (in-place)."""
        # Modify the original message's content directly (no copy)
        content = message.content
        if isinstance(content, list):
            for i, item in enumerate(content):
                try:
                    # Use model_validate for type-safe handling
                    tool_use = ant.MessageContentToolUse.model_validate(item)
                    if tool_use.id == tool_use_id:
                        tool_use.input = new_input
                        content[i] = tool_use.model_dump(mode="json", exclude_none=True)
                        break
                except Exception:
                    # Not a tool_use or validation failed, skip to next item
                    continue

    def _compact_tool_result(
        self,
        message: CfMessage,
        tool_use_id: str,
        new_content: str,
    ) -> None:
        """Replace tool result content with compacted version (in-place)."""
        # Modify the original message's content directly (no copy)
        content = message.content
        if isinstance(content, list):
            for i, item in enumerate(content):
                try:
                    # Use model_validate for type-safe handling
                    tool_result = ant.MessageContentToolResult.model_validate(item)
                    if tool_result.tool_use_id == tool_use_id:
                        tool_result.content = new_content
                        content[i] = tool_result.model_dump(
                            mode="json", exclude_none=True
                        )
                        break
                except Exception:
                    # Not a tool_result or validation failed, skip to next item
                    continue

    async def _ignore_tool_use_messages(
        self,
        tool_use_msgs: _ToolUseMessages,
        tool_use_id: str,
        context: AnalectRunContext,
        placeholder_text: str | None = None,
    ) -> None:
        """Clear a tool use (used internally for context_edit tool cleanup).

        Replaces tool_use input with cleared marker and tool_result with placeholder text.
        Deletes tool_use_info message.

        Args:
            tool_use_msgs: The message references from the map
            tool_use_id: The tool use ID
            context: The analect run context
            placeholder_text: Optional custom placeholder text
        """
        tool_use = tool_use_msgs.tool_use
        if tool_use is not None:
            self._compact_tool_use_input(
                tool_use,
                tool_use_id,
                {CLEARED_TOOL_USE_KEY: True},
            )
            self._ignored_tool_use_ids.add(tool_use_id)

        tool_result = tool_use_msgs.tool_result
        if tool_result is not None:
            self._compact_tool_result(
                tool_result,
                tool_use_id,
                placeholder_text or "The result was cleared",
            )
            self._ignored_tool_use_ids.add(tool_use_id)

        tool_use_info_msg = tool_use_msgs.tool_use_info
        if tool_use_info_msg is not None:
            context.memory_manager.delete_message(tool_use_info_msg)

        self._tool_use_message_map.pop(tool_use_id, None)

    async def _rewrite_tool_use_messages(
        self,
        tool_use_msgs: _ToolUseMessages,
        edit: ContextEdit,
        context: AnalectRunContext,
    ) -> None:
        """Rewrite tool use messages with replacement content (in-place).

        Marks tool_use input with rewritten marker, replaces tool_result content,
        and deletes tool_use_info.

        Args:
            tool_use_msgs: The message references from the map
            edit: The context edit with replacement values
            context: The analect run context
        """
        tool_use = tool_use_msgs.tool_use
        if tool_use is not None:
            if edit.tool_use_input_replacement is not None:
                new_input = {REWRITTEN_TOOL_USE_KEY: True, **edit.tool_use_input_replacement}
            else:
                # Extract original input and add rewritten marker
                original_input = self._extract_tool_use_input(tool_use, edit.tool_use_id)
                new_input = {REWRITTEN_TOOL_USE_KEY: True, **(original_input or {})}
            self._compact_tool_use_input(tool_use, edit.tool_use_id, new_input)

        tool_result = tool_use_msgs.tool_result
        if tool_result is not None and edit.tool_result_content_replacement is not None:
            self._compact_tool_result(
                tool_result,
                edit.tool_use_id,
                edit.tool_result_content_replacement,
            )

        # Delete tool_use_info (system metadata)
        tool_use_info_msg = tool_use_msgs.tool_use_info
        if tool_use_info_msg is not None:
            context.memory_manager.delete_message(tool_use_info_msg)

    @override
    async def on_session_complete(self, context: AnalectRunContext) -> None:
        """Write token usage and session-end training data with outcome annotations."""
        # Always call super to write token_usage.json
        await super().on_session_complete(context)

        # Write compression agent costs to separate file
        if self._compression_agent_usage:
            total_input = sum(u["input_tokens"] for u in self._compression_agent_usage)
            total_output = sum(u["output_tokens"] for u in self._compression_agent_usage)

            # Compute monetary cost using same pricing as main agent
            pricing = MODEL_PRICING.get(self.pricing_model, MODEL_PRICING[DEFAULT_PRICING_MODEL])
            tokens_per_million = 1_000_000
            # Subagent calls are never cached — all input is uncached
            total_input_cost_usd = total_input * pricing["input"] / tokens_per_million
            total_output_cost_usd = total_output * pricing["output"] / tokens_per_million
            total_cost_usd = total_input_cost_usd + total_output_cost_usd

            subagent_data = {
                "num_calls": len(self._compression_agent_usage),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "pricing_model": self.pricing_model,
                "total_input_cost_usd": total_input_cost_usd,
                "total_output_cost_usd": total_output_cost_usd,
                "total_cost_usd": total_cost_usd,
                "per_call": self._compression_agent_usage,
            }
            try:
                with open("token_usage_subagent.json", "w") as f:
                    json.dump(subagent_data, f, indent=2)
            except Exception as e:
                context_edit_logger.warning("Failed to write token_usage_subagent.json: %s", e)

        if self.training_data_dir is None:
            return

        # Build set of all edited tool_use_ids
        edited_ids: set[str] = set()
        for edit_event in self._edit_history:
            for edit in edit_event.get("edits", []):
                edited_ids.add(edit.get("tool_use_id", ""))

        # Annotate each tool use with outcome
        tool_use_outcomes = []
        for tuid, registry_entry in self._tool_use_registry.items():
            if registry_entry.get("tool_name") == CONTEXT_EDIT_TOOL_NAME:
                continue
            was_edited = tuid in edited_ids
            was_rewritten = tuid in self._ignored_tool_use_ids
            tool_use_outcomes.append({
                "tool_use_id": tuid,
                "tool_name": registry_entry.get("tool_name", "unknown"),
                "created_at_turn": registry_entry.get("created_at_turn", -1),
                "input_tokens": registry_entry.get("input_tokens", 0),
                "result_tokens": registry_entry.get("result_tokens", 0),
                "image_count": registry_entry.get("image_count", 0),
                "was_edited": was_edited,
                "was_rewritten": was_rewritten,
            })

        # Collect token usage from the estimator
        call_history = self.get_call_history()
        total_input = sum(c.get("input_tokens", 0) for c in call_history)
        total_output = sum(c.get("output_tokens", 0) for c in call_history)
        total_cache_read = sum(c.get("cache_read_tokens", 0) for c in call_history)

        # Count meta-cognitive overhead (tokens spent on context_edit tool calls)
        meta_cognitive_output_tokens = 0
        for edit_event in self._edit_history:
            # Each context_edit tool call costs output tokens for the model's reasoning
            # We approximate this from the edit count (the model reasons about each edit)
            meta_cognitive_output_tokens += len(edit_event.get("edits", [])) * 100  # rough estimate

        self._log_training_data("edit_outcomes.jsonl", {
            "event": "session_complete",
            "total_turns": self._turn_counter,
            "total_tool_uses": len(self._tool_use_registry),
            "total_edits_applied": sum(len(e.get("edits", [])) for e in self._edit_history),
            "total_edit_events": len(self._edit_history),
            "total_tokens_saved_by_edits": sum(e.get("tokens_saved", 0) for e in self._edit_history),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "num_llm_calls": len(call_history),
            "tool_use_outcomes": tool_use_outcomes,
            "edit_history": self._edit_history,
        })
