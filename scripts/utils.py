import sys
import traceback
from typing import Optional

from confucius.analects.code.entry import CodeAssistEntry, SmartContextConfig
from confucius.core.entry.base import EntryInput
from confucius.core.entry.entry import Entry
from confucius.lib.confucius import Confucius


async def run_agent_with_prompt(
    prompt: str,
    entry_name: str = "Code",
    verbose: bool = False,
    smart_context_enabled: bool = False,
    smart_context_compression_threshold: Optional[int] = None,
    smart_context_clear_at_least: Optional[int] = None,
    smart_context_clear_at_least_tolerance: Optional[float] = None,
    smart_context_reminder_enabled: Optional[bool] = None,
    smart_context_reminder_ratio: Optional[float] = None,
    cache_min_prompt_length: Optional[int] = None,
    cache_max_num_checkpoints: Optional[int] = None,
    context_edit_log_dir: Optional[str] = None,
    smart_context_enable_context_usage: bool = False,
    smart_context_context_window_size: Optional[int] = None,
    compression_agent_enabled: bool = False,
    compression_agent_model: Optional[str] = None,
    compression_agent_max_tokens: Optional[int] = None,
    compression_cooldown_tokens: Optional[int] = None,
    max_edits_per_call: Optional[int] = None,
    architect_trigger_tokens: Optional[int] = None,
    architect_min_prompt_length: Optional[int] = None,
) -> None:
    """
    Run the Confucius Code agent with a given prompt and wait for completion.

    Args:
        prompt: The input prompt to send to the agent
        entry_name: Name of the entry to run (default: "Code")
        verbose: Enable verbose logging
        smart_context_enabled: If True, enables SmartContextManagementExtension
            and disables Anthropic caching and LLMCodingArchitectExtension
        smart_context_compression_threshold: Token count that triggers context
            optimization enforcement (maps to input_tokens_trigger)
        smart_context_clear_at_least: Minimum tokens that must be cleared per
            edit to justify the operation
        smart_context_clear_at_least_tolerance: Tolerance for enforcing the
            clear_at_least threshold (default 0.75)
        smart_context_reminder_enabled: Whether to inject the soft reminder
            SystemMessage when approaching the trigger threshold
        smart_context_reminder_ratio: Ratio of input_tokens_trigger at which to
            start showing reminders (only used when reminder_enabled=True)
        cache_min_prompt_length: Minimum token length between cache breakpoints
        cache_max_num_checkpoints: Maximum number of cache breakpoints
        context_edit_log_dir: Directory to write context_edits.jsonl log file.
            Only meaningful when smart_context_enabled is True.
        smart_context_enable_context_usage: Whether to include cumulative context
            usage in system_info tags after tool use.
        smart_context_context_window_size: Total context window size in tokens
            for usage percentage calculation.
    """
    cf: Confucius = Confucius(verbose=verbose)

    try:
        # SmartContextConfig is specific to CodeAssistEntry, so we check for "Code" entry.
        # For other entries, smart_context_enabled is ignored and we use the standard Entry dispatcher.
        # We also need to directly instantiate CodeAssistEntry when cache parameters are provided.
        if entry_name == "Code" and (smart_context_enabled or cache_min_prompt_length is not None or cache_max_num_checkpoints is not None or architect_trigger_tokens is not None):
            # Directly instantiate CodeAssistEntry with custom configuration
            smart_context_config = SmartContextConfig(
                enabled=smart_context_enabled,
                compression_threshold=smart_context_compression_threshold,
                clear_at_least=smart_context_clear_at_least,
                clear_at_least_tolerance=smart_context_clear_at_least_tolerance,
                reminder_enabled=smart_context_reminder_enabled,
                reminder_ratio=smart_context_reminder_ratio,
                cache_min_prompt_length=cache_min_prompt_length,
                cache_max_num_checkpoints=cache_max_num_checkpoints,
                log_dir=context_edit_log_dir,
                enable_context_usage=smart_context_enable_context_usage,
                context_window_size=smart_context_context_window_size,
                compression_agent_enabled=compression_agent_enabled,
                compression_agent_model=compression_agent_model,
                compression_agent_max_tokens=compression_agent_max_tokens,
                compression_cooldown_tokens=compression_cooldown_tokens,
                max_edits_per_call=max_edits_per_call,
                architect_trigger_tokens=architect_trigger_tokens,
                architect_min_prompt_length=architect_min_prompt_length,
            )
            code_entry = CodeAssistEntry(smart_context_config=smart_context_config)
            await cf.invoke_analect(code_entry, EntryInput(question=prompt))
        else:
            # Use Entry with EntryInput to run the specified entry
            await cf.invoke_analect(
                Entry(), EntryInput(question=prompt, entry_name=entry_name)
            )

    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)
        print(f"Stack trace:\n{traceback.format_exc()}", file=sys.stderr)
        raise
    finally:
        # Dump message trajectory after completion
        cf.dump_trajectory()

        # Save session state like the REPL does
        await cf.save(raise_exception=False)
