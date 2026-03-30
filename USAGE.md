# Usage Guide: Smart Context Management & Cache Configuration

## Overview

This guide shows how to run the script with the new smart context management and cache configuration parameters.

---

## Basic Usage

### 1. Default Run (No Smart Context)

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  > logs.txt
```

**Behavior:**
- LLMCodingArchitectExtension included (automatic planning)
- AnthropicPromptCaching with default settings (min_prompt_length=10000, max_num_checkpoints=4)
- No smart context management

---

## Smart Context Management

### 2. Enable Smart Context (Default Reminder Settings)

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 60000 \
  --clear-at-least 10000 \
  --clear-at-least-tolerance 0.75 \
  > logs.txt
```

**Behavior:**
- Smart context management enabled
- Reminder at 48,000 tokens (60K × 0.8 default ratio)
- Enforcement at 60,000 tokens
- Must clear at least 10,000 tokens per edit
- LLMCodingArchitectExtension excluded (no automatic planning)

### 3. Enable Smart Context with Custom Reminder Ratio

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 100000 \
  --reminder-ratio 0.9 \
  --clear-at-least 50000 \
  > logs.txt
```

**Behavior:**
- Reminder at 90,000 tokens (100K × 0.9)
- Enforcement at 100,000 tokens
- 10% warning buffer before enforcement

### 4. Enable Smart Context WITHOUT Reminder (Enforcement Only)

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 100000 \
  --disable-reminder \
  --clear-at-least 50000 \
  > logs.txt
```

**Behavior:**
- No reminder message
- Enforcement at 100,000 tokens
- Agent gets comprehensive enforcement message with all guidance
- No advance warning

---

## Cache Configuration

### 5. Configure Cache Parameters (Without Smart Context)

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --cache-min-prompt-length 15000 \
  --cache-max-num-checkpoints 3 \
  > logs.txt
```

**Behavior:**
- Default mode (no smart context)
- Cache breakpoints every 15K tokens (instead of default 10K)
- Maximum 3 cache breakpoints (instead of default 4)
- LLMCodingArchitectExtension included

### 6. Smart Context with Custom Cache Parameters

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 80000 \
  --clear-at-least 40000 \
  --cache-min-prompt-length 12000 \
  --cache-max-num-checkpoints 5 \
  > logs.txt
```

**Behavior:**
- Smart context enabled with reminder at 64K, enforcement at 80K
- Custom cache configuration
- Cache breakpoints every 12K tokens, max 5 breakpoints

---

## Complete Parameter Reference

### Smart Context Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable-smart-context` | flag | False | Enable SmartContextManagementExtension |
| `--compression-threshold` | int | None | Token count triggering enforcement (maps to input_tokens_trigger) |
| `--clear-at-least` | int | None | Minimum tokens to clear per edit |
| `--clear-at-least-tolerance` | float | None | Tolerance for clear_at_least threshold (default 0.75) |
| `--disable-reminder` | flag | False | Disable soft reminder message |
| `--reminder-ratio` | float | None | Ratio for reminder trigger (default 0.8) |

### Cache Parameters (Work with or without Smart Context)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cache-min-prompt-length` | int | None | Minimum tokens between cache breakpoints (default 10000) |
| `--cache-max-num-checkpoints` | int | None | Maximum cache breakpoints (default 4) |

### General Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--prompt` | string | Path to prompt text file (required) |
| `--verbose` | flag | Enable verbose logging |

---

## Common Scenarios

### Scenario 1: Long-Running Tasks (High Token Usage Expected)

**Goal:** Maximize context window usage, get early warning

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 120000 \
  --reminder-ratio 0.75 \
  --clear-at-least 60000 \
  > logs.txt
```

- Reminder at 90K tokens (120K × 0.75)
- Enforcement at 120K tokens
- 30K token warning buffer

### Scenario 2: Aggressive Context Management

**Goal:** Start managing context early, frequent optimizations

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 50000 \
  --reminder-ratio 0.7 \
  --clear-at-least 20000 \
  > logs.txt
```

- Reminder at 35K tokens (50K × 0.7)
- Enforcement at 50K tokens
- Lower clear_at_least for more frequent edits

### Scenario 3: Strict Enforcement, No Warnings

**Goal:** No gradual warnings, immediate enforcement

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 100000 \
  --disable-reminder \
  --clear-at-least 50000 \
  > logs.txt
```

- No reminder
- Enforcement at 100K tokens
- Agent gets comprehensive enforcement message

### Scenario 4: Fine-Tuned Caching

**Goal:** Optimize cache performance for specific use case

```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --cache-min-prompt-length 20000 \
  --cache-max-num-checkpoints 6 \
  > logs.txt
```

- No smart context (default behavior)
- Fewer, larger cache breakpoints
- Good for long, stable conversations

---

## Migration Guide

### From Old Command

**Before (your previous command):**
```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 60000 \
  --clear-at-least 10000 \
  --clear-at-least-tolerance 0.75
```

**After (equivalent with new parameters):**
```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 60000 \
  --clear-at-least 10000 \
  --clear-at-least-tolerance 0.75
  # Optional new parameters:
  # --reminder-ratio 0.8 (default, shows reminder at 48K)
  # --cache-min-prompt-length 10000 (default)
  # --cache-max-num-checkpoints 4 (default)
```

**Behavior comparison:**
- **Old:** Reminder at 60K, enforcement at 72K (60K × 1.2)
- **New:** Reminder at 48K (60K × 0.8), enforcement at 60K

**To get similar timing to old behavior:**
```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 72000 \
  --reminder-ratio 0.833 \
  --clear-at-least 10000 \
  --clear-at-least-tolerance 0.75
```

This gives reminder at ~60K, enforcement at 72K.

---

## Understanding Reminder vs Enforcement

### When reminder_enabled=True (default)

```
Token Usage Timeline:
0K ──────── reminder_trigger ──────── enforcement_trigger ──────>
            (trigger × ratio)         (trigger)
            
Example (trigger=100K, ratio=0.8):
0K ──────── 80K ──────── 100K ──────>
            ↓            ↓
         REMINDER    ENFORCEMENT
       (detailed)    (brief/detailed)
```

**At reminder threshold (80K):**
- Comprehensive 46-line message
- Token counts, guidance, constraints
- Agent can still use all tools
- Soft warning to optimize soon

**At enforcement threshold (100K):**
- Brief or comprehensive message (depending on scenario)
- Blocks all tools except memory and context_edit
- Agent MUST optimize before continuing

### When reminder_enabled=False (--disable-reminder)

```
Token Usage Timeline:
0K ──────────────────────── enforcement_trigger ──────>
                            (trigger)
            
Example (trigger=100K):
0K ──────────────────────── 100K ──────>
                            ↓
                       ENFORCEMENT
                     (comprehensive)
```

**At enforcement threshold (100K):**
- Comprehensive 33-line message
- Includes all guidance (same as reminder)
- Blocks tools immediately
- No advance warning

---

## Tips & Best Practices

### 1. Choosing compression-threshold
- **Development/Testing:** 50K-80K (aggressive)
- **Production/Long Tasks:** 100K-150K (conservative)
- **Very Long Tasks:** 150K+ (maximize window)

### 2. Choosing reminder-ratio
- **Early Warning:** 0.7-0.75 (25-30% buffer)
- **Standard:** 0.8 (default, 20% buffer)
- **Late Warning:** 0.9 (10% buffer)
- **Disable:** Use `--disable-reminder` for no warning

### 3. Choosing clear-at-least
- Should be meaningful relative to compression-threshold
- **Rule of thumb:** 30-50% of compression-threshold
- **Example:** threshold=100K → clear-at-least=40K-50K
- Prevents ineffective micro-optimizations

### 4. Cache Configuration
- **Default (10K, 4 checkpoints):** Good for most cases
- **Larger chunks (15K-20K, 3-4 checkpoints):** Stable conversations
- **More checkpoints (8K, 6 checkpoints):** Frequently changing context
- **Trade-off:** More checkpoints = more cache hits, but smaller chunks

### 5. When to Disable Reminder
- When you want strict enforcement at exact threshold
- When advance warnings are not useful
- Testing enforcement behavior
- Short-lived sessions where early optimization isn't needed

---

## Troubleshooting

### Issue: Agent keeps hitting enforcement
**Solution:** Lower compression-threshold or increase reminder-ratio

### Issue: Too many reminder messages
**Solution:** Increase compression-threshold or disable reminder

### Issue: Context edits not saving enough tokens
**Solution:** Lower clear-at-least or adjust tolerance

### Issue: High cache costs
**Solution:** Increase cache-min-prompt-length to create fewer breakpoints

### Issue: Poor cache hit rate
**Solution:** Decrease cache-min-prompt-length or increase max-num-checkpoints

---

## Status Display

When running the script, you'll see configuration printed:

```
Running Confucius Code Agent with prompt from file: /tmp/task.txt
SmartContextManagementExtension is ENABLED
  - Anthropic caching: ON
  - LLMCodingArchitectExtension: EXCLUDED
  - compression_threshold: 60000
  - clear_at_least: 10000
  - clear_at_least_tolerance: 0.75
  - reminder: ENABLED
  - reminder_ratio: 0.8
Cache configuration:
  - min_prompt_length: 12000
  - max_num_checkpoints: 5
```

This confirms your configuration is active.

---

## Summary

**Key Changes from Previous Version:**

1. ✅ `compression-threshold` is now the enforcement point (not reminder point)
2. ✅ Reminder fires earlier at `threshold × ratio` (default 0.8)
3. ✅ Can disable reminder with `--disable-reminder`
4. ✅ Can customize reminder timing with `--reminder-ratio`
5. ✅ Cache parameters always available (with or without smart context)
6. ✅ Enforcement message is comprehensive when reminder disabled

**Your Command Still Works:**
```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 60000 \
  --clear-at-least 10000 \
  --clear-at-least-tolerance 0.75
```

Just note: Now reminder appears at 48K and enforcement at 60K (instead of 60K and 72K).
