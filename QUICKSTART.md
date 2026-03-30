# Quick Reference: Command Examples

## Your Original Command

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

**This still works!** 

**New behavior:**
- Reminder at 48,000 tokens (60K × 0.8)
- Enforcement at 60,000 tokens
- More warning time before enforcement

---

## Most Common Usage Patterns

### 1. Basic (No Smart Context)
```bash
python /usr/local/bin/app.pex --prompt /tmp/task.txt --verbose > logs.txt
```

### 2. Smart Context with Default Settings
```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 100000 \
  --clear-at-least 50000 \
  > logs.txt
```

### 3. Smart Context with Early Reminder
```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 100000 \
  --reminder-ratio 0.7 \
  --clear-at-least 50000 \
  > logs.txt
```
*Reminder at 70K, enforcement at 100K*

### 4. Smart Context WITHOUT Reminder
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
*No reminder, enforcement at 100K with full guidance*

### 5. Configure Cache Only
```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --cache-min-prompt-length 15000 \
  --cache-max-num-checkpoints 3 \
  > logs.txt
```
*No smart context, just custom cache settings*

### 6. Everything Combined
```bash
python /usr/local/bin/app.pex \
  --prompt /tmp/task.txt \
  --verbose \
  --enable-smart-context \
  --compression-threshold 120000 \
  --reminder-ratio 0.75 \
  --clear-at-least 60000 \
  --cache-min-prompt-length 12000 \
  --cache-max-num-checkpoints 5 \
  > logs.txt
```

---

## Parameter Quick Reference

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `--enable-smart-context` | off | Turn on smart context management |
| `--compression-threshold N` | - | Enforcement at N tokens |
| `--reminder-ratio 0.X` | 0.8 | Reminder at N × 0.X tokens |
| `--disable-reminder` | off | Skip reminder, enforce immediately |
| `--clear-at-least N` | - | Must clear N tokens per edit |
| `--clear-at-least-tolerance 0.X` | 0.75 | Tolerance for clearing |
| `--cache-min-prompt-length N` | 10000 | Tokens between cache points |
| `--cache-max-num-checkpoints N` | 4 | Max cache breakpoints |
| `--prompt FILE` | required | Path to prompt file |
| `--verbose` | off | Verbose logging |

---

## Reminder Math

| threshold | ratio | reminder_at | enforcement_at |
|-----------|-------|-------------|----------------|
| 60000 | 0.8 (default) | 48000 | 60000 |
| 100000 | 0.8 (default) | 80000 | 100000 |
| 100000 | 0.7 | 70000 | 100000 |
| 100000 | 0.9 | 90000 | 100000 |
| 100000 | disabled | - | 100000 |

---

## What Changed from Previous Version?

### OLD Model
- `compression-threshold` = reminder point
- Enforcement at threshold × 1.2
- Example: threshold=60K → reminder@60K, enforce@72K

### NEW Model  
- `compression-threshold` = enforcement point
- Reminder at threshold × ratio (default 0.8)
- Example: threshold=60K → reminder@48K, enforce@60K

### Migration
Your old command still works but timing changed:

**Old:** reminder@60K, enforce@72K  
**New:** reminder@48K, enforce@60K

To keep old timing:
```bash
--compression-threshold 72000 --reminder-ratio 0.833
```
This gives: reminder@60K, enforce@72K

---

## Need More Help?

See full documentation: [USAGE.md](USAGE.md)
