# SWE-Bench Pro Batch Evaluation with Confucius

This document describes how to run the Confucius Code Agent at scale on
[SWE-Bench Pro](https://www.swebench.com/) problems using the batch evaluation
script at `scripts/run_batch_swebench.py`.

---

## Overview

The script automates the agent run loop:

1. **Load problems** from `sbp-problems.jsonl` (a JSON array of SWE-Bench Pro problem definitions).
2. **For each problem × experiment config**:
   - Pull/start a **podman container** using the problem's Docker image.
   - Inject the problem statement as `/tmp/task.txt` inside the container.
   - Execute the **Confucius Code Agent** (`app.pex`) inside the container.
   - Extract the resulting **git diff** (the agent's patch).
   - Copy all artefacts (patch, logs, metadata) to a local results directory.
   - **Stop and remove** the container (even on failure).
3. Optionally **upload results** to Manifold.
4. Print a **summary report** of succeeded / failed runs.

Two experiment configurations are run by default (A/B):

| Config | Description |
|---|---|
| `no_compression` | Runs the agent with default settings (no smart context) |
| `smart_context` | Runs the agent with `--enable-smart-context` and configurable thresholds |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **podman** (or docker) | Must be installed and accessible on `$PATH`. `docker` can be used via `--container_runtime docker`. |
| **`with-proxy`** | The `with-proxy` helper must be on `$PATH` (available on devvm). It wraps the podman run command for correct network/proxy routing. |
| **SWE-Bench Pro images** | Images are pulled from the internal registry `vmvm-registry.fbinfra.net/sweap_retag/<instance_id>` on demand. |
| **`cf_env.tar.gz`** | Conda environment tarball placed in `--data_dir` (default `~/workspace/sbp_memory`). Unpacked into `/opt/appenv` inside the container. |
| **`app.pex`** | The Confucius PEX binary placed in `--data_dir`. Copied to `/usr/local/bin/app.pex` inside the container by the setup step. |
| **`sbp-problems.jsonl`** | Included in the repo root. Contains SWE-Bench Pro problem definitions. |
| **AWS credentials** | Required if using AWS Bedrock as the LLM provider. Obtained via `cloud aws get-creds`. |
| **`manifold` CLI** | Required only if you want to upload results (`--manifold_base_path`). |

---

## How the podman container lifecycle works

For each `(instance_id, config)` pair the script:

1. **Derives the image name** from the problem record:
   - Uses the `image_name` field if present in the JSONL record.
   - Otherwise constructs `{image_registry}/{instance_id}` (default registry: `vmvm-registry.fbinfra.net/sweap_retag`).
2. **Starts a detached container** via `with-proxy podman run -d` with a deterministic name
   `cf_sweb_<instance_id>_<config>` (special characters replaced with `_`).
   The container is started with:
   - `--network host` and proxy environment variables for outbound connectivity.
   - `--memory=4g --memory-swap=4g` and `--userns=host` for resource limits.
   - `-v <data_dir>:/data` to supply `cf_env.tar.gz` and `app.pex`.
   - AWS credential and region environment variables passed from the host.
   - `TASK_ID` and `PROBLEM_STATEMENT` environment variables.
3. **Bootstraps the environment** inside the container via `podman exec`:
   - Unpacks `cf_env.tar.gz` into `/opt/appenv` and runs `conda-unpack`.
   - Copies `app.pex` to `/usr/local/bin/`.
4. **Writes the prompt** to `/tmp/task.txt` inside the running container via `podman exec`.
5. **Runs the agent** via `podman exec -w /app … python /usr/local/bin/app.pex --prompt /tmp/task.txt --cache-min-prompt-length 0 --verbose [flags]`.
6. **Extracts result artifacts** from the container:
   - `patch.diff` — committed changes via `git show --format= --no-color`
   - `logs.txt` — agent log at `/app/logs.txt`
   - `context_edits.jsonl` — context edit log at `/app/context_edits.jsonl` (smart_context config only)
   - `traj_*.json` — trajectory files at `/tmp/confucius/traj_*.json`
   - `token_usage.json` — token usage at `/app/token_usage.json`
7. **Writes result files** to `<result_dir>/<timestamp>/<config>/<instance_id>/`.
8. **Stops and removes** the container (`podman stop && podman rm -f`) in a `finally` block so cleanup always happens.

The `--rm` flag is also passed to `podman run` so that if the container exits on its own it is removed automatically.

---

## Results directory structure

```
results_swebench/
└── 20240315T120000Z/          # timestamp of the batch run
    ├── batch_run.log           # combined log file (all problems, all configs) — local only, not uploaded
    ├── summary.json            # machine-readable results for every (problem, config) pair — local only, not uploaded
    ├── no_compression/
    │   └── <instance_id>/
    │       ├── patch.diff          # committed patch from git show
    │       ├── logs.txt            # agent log file
    │       ├── traj_*.json         # trajectory file(s)
    │       ├── token_usage.json    # token usage statistics
    │       ├── agent_output.log    # stdout + stderr from the agent run
    │       └── metadata.json       # timing, exit code, image, error (if any)
    └── smart_context_50000_20000/  # name encodes --compression_threshold and --clear_at_least
        └── <instance_id>/
            ├── patch.diff
            ├── logs.txt
            ├── traj_*.json
            ├── token_usage.json
            ├── context_edits.jsonl # context edit log (smart_context only)
            ├── agent_output.log
            └── metadata.json
```

### `metadata.json` fields

| Field | Description |
|---|---|
| `instance_id` | SWE-Bench Pro instance identifier |
| `config` | Experiment configuration (`no_compression` or `smart_context`) |
| `image` | Container image used |
| `container_name` | Name given to the podman container |
| `started_at` | ISO-8601 UTC timestamp when the problem started |
| `finished_at` | ISO-8601 UTC timestamp when the problem finished |
| `success` | `true` if the agent exited with code 0 |
| `exit_code` | Agent exit code (`-1` for timeout) |
| `duration_seconds` | Wall-clock seconds for the full run |
| `error` | Error message if an exception was raised (otherwise `null`) |

---

## Manifold upload structure

When `--manifold_base_path` is set (default: `confucius/tree/slin_test/swebench-pro-evals`),
each result directory is uploaded as:

```
<manifold_base_path>/<config>/<instance_id>/
```

Upload uses the `manifold put` CLI with automatic retry (up to `--max_retries` attempts).

---

## GitHub upload

Results are uploaded to the GitHub repository specified by `--github_results_repo`
(default: `samuellin01/memory_experiments`) **incrementally after each task completes**,
rather than at the end of the batch. This means partial results are preserved even if the
batch is interrupted (crash, devvm restart, etc.).

For example:
```
local:  results_swebench/20240315T120000Z/no_compression/instance_id/patch.diff
GitHub: sbp/instance_id/no_compression/patch.diff

local:  results_swebench/20240315T120000Z/smart_context_50000_20000/instance_id/patch.diff
GitHub: sbp/instance_id/smart_context_50000_20000/patch.diff
```

Results are grouped by task (instance) first and then by policy (config) in the target
repository, so all runs for a given problem are co-located under one folder.

The numbers in `smart_context_50000_20000` come from `--compression_threshold` (default 50000)
and `--clear_at_least` (default 20000), so runs with different parameters are kept separate.

> **Note:** `batch_run.log` and `summary.json` at the root of the result directory are
> **local-only** files and are **not** uploaded to GitHub.

### Requirements

- Set the `GITHUB_TOKEN` environment variable to a fine-grained PAT with **Contents read/write**
  permission on the target repository (`samuellin01/memory_experiments` by default).
- The upload is routed through the devvm proxy (`http://127.0.0.1:18080`) automatically.
- If `GITHUB_TOKEN` is not set, a warning is logged and the upload is skipped (the eval results
  are still saved locally).
- Individual file upload failures are logged and skipped; remaining files continue to upload.
- Files larger than 50 MB are skipped (GitHub Contents API limit).

### Skip the upload

```bash
python scripts/run_batch_swebench.py --skip_github_upload
```

### Upload to a different repo or path

```bash
python scripts/run_batch_swebench.py \
  --github_results_repo myorg/my-results-repo \
  --github_results_path experiments/sbp
```

---

## Parameter reference

| Argument | Default | Description |
|---|---|---|
| `--problems_file` | `sbp-problems.jsonl` | Path to the problems file (JSON array or JSONL) |
| `--problem_ids` | _(all)_ | Space-separated list of specific instance IDs to run |
| `--configs` | `no_compression smart_context` | Experiment configurations to run |
| `--compression_threshold` | `50000` | Token count that triggers smart context compression |
| `--clear_at_least` | `20000` | Minimum tokens cleared per compression operation |
| `--clear_at_least_tolerance` | `0.75` | Tolerance for clear-at-least compression |
| `--pex_path` | `/usr/local/bin/app.pex` | Path to the `app.pex` binary inside the container |
| `--container_runtime` | `podman` | Container runtime (`podman` or `docker`) |
| `--image_registry` | `vmvm-registry.fbinfra.net/sweap_retag` | Container image registry prefix (instance_id is appended) |
| `--data_dir` | `~/workspace/sbp_memory` | Host directory mounted as `/data` in the container |
| `--container_memory` | `4g` | Memory limit per container (passed to `--memory` and `--memory-swap`) |
| `--result_dir` | `./results_swebench` | Local directory for storing results |
| `--task_timeout` | `3600` | Per-problem timeout in seconds |
| `--max_retries` | `3` | Maximum retry attempts per problem |
| `--manifold_base_path` | `confucius/tree/slin_test/swebench-pro-evals` | Base Manifold path for uploads |
| `--upload_timeout` | `3600` | Timeout per Manifold upload attempt (seconds) |
| `--skip_credential_refresh` | `false` | Skip AWS credential refresh |
| `--credential_refresh_interval` | `0` | Refresh credentials every N seconds (0 = once at start) |
| `--dry_run` | `false` | Print commands without executing them |
| `--github_results_repo` | `samuellin01/memory_experiments` | Target GitHub repo (`owner/name`) to upload results to |
| `--github_results_path` | `sbp` | Base folder in the target GitHub repo; results land at `{github_results_path}/` |
| `--skip_github_upload` | `false` | Skip uploading results to GitHub after the run |

---

## Example commands

### Run all problems with both configs
```bash
cd /path/to/sbp_memory
python scripts/run_batch_swebench.py
```

### Run a specific subset of problems
```bash
python scripts/run_batch_swebench.py \
  --problem_ids instance_nodebb__nodebb-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan \
               instance_ansible__ansible-f327e65d11bb905ed9f15996024f857a95592629-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
```

### Run only the `smart_context` config
```bash
python scripts/run_batch_swebench.py --configs smart_context
```

### Dry-run to preview what would be executed
```bash
python scripts/run_batch_swebench.py --dry_run --problem_ids <instance_id>
```

### Use a custom problems file
```bash
python scripts/run_batch_swebench.py --problems_file /path/to/my-problems.jsonl
```

### Use docker instead of podman
```bash
python scripts/run_batch_swebench.py --container_runtime docker
```

### Skip credential refresh (e.g., credentials already set in environment)
```bash
python scripts/run_batch_swebench.py --skip_credential_refresh
```

### Custom thresholds for smart context
```bash
python scripts/run_batch_swebench.py \
  --configs smart_context \
  --compression_threshold 80000 \
  --clear_at_least 40000 \
  --clear_at_least_tolerance 0.75
```

### Custom data directory and image registry
```bash
python scripts/run_batch_swebench.py \
  --data_dir /my/workspace/sbp_memory \
  --image_registry my-registry.example.com/images
```

### Custom result directory and manifold path
```bash
python scripts/run_batch_swebench.py \
  --result_dir /tmp/my_swebench_results \
  --manifold_base_path confucius/tree/myuser/swebench-pro-evals
```

### Run with GitHub upload (default)
```bash
export GITHUB_TOKEN="github_pat_your_token_here"
python scripts/run_batch_swebench.py
# Results are uploaded to samuellin01/memory_experiments under sbp/<timestamp>/
```

### Run and skip GitHub upload
```bash
python scripts/run_batch_swebench.py --skip_github_upload
```

---

## Troubleshooting

### Podman issues

**`Error: image not found`**

The container image for a problem has not been pulled from the registry. Pull it manually:
```bash
with-proxy podman pull vmvm-registry.fbinfra.net/sweap_retag/<instance_id>
```

To use a different registry, pass `--image_registry`:
```bash
python scripts/run_batch_swebench.py --image_registry my-registry.example.com/images
```

**`Error: container name already in use`**

A previous run left a container running. Remove it:
```bash
podman rm -f cf_sweb_<instance_id>_<config>
```

Or list all running containers:
```bash
podman ps -a | grep cf_sweb_
```

**Permission errors with podman**

On some systems podman requires a rootless setup. Consult the [podman rootless guide](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md).

### Credential issues

**`'cloud' CLI not found; skipping credential refresh.`**

The `cloud aws get-creds` command is not available. Either:
- Pass `--skip_credential_refresh` and set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` manually.
- Install the internal `cloud` CLI if available in your environment.

**`AWS credential refresh returned exit code 2` / missing required arguments**

The script runs the following command to refresh credentials:
```bash
cloud aws get-creds 009160068926 --role SSOAdmin --duration 14400
```
It then parses the `export VAR=value` lines from the output and applies them to the current process environment so subsequent subprocess calls (manifold uploads, etc.) inherit valid credentials. Credentials are refreshed once at the start and again before every SWE-Bench problem task.

**`AWS credential refresh returned exit code 1`**

Check that you are authenticated with the `cloud` CLI (`cloud login`) and have the necessary permissions.

**AWS API errors inside the container (`NoCredentialProviders`, `NoRegionError`, etc.)**

The batch script automatically passes AWS region environment variables into each container:
```
AWS_REGION_NAME=us-west-2
AWS_DEFAULT_REGION=us-west-2
AWS_REGION=us-west-2
```
It also forwards the current host credential env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`) into every `podman exec` call so the agent has valid credentials. If you are seeing region or credential errors, verify that the host credentials are valid (run `cloud aws get-creds 009160068926 --role SSOAdmin --duration 14400` manually) before starting the batch run.

### Container extraction issues

**Agent timeout (`exit_code=-1`)**

The agent exceeded `--task_timeout` seconds. Increase the timeout or investigate why the problem takes so long:
```bash
python scripts/run_batch_swebench.py --task_timeout 7200
```

**Empty `patch.diff`**

The agent ran but did not make any code changes, or the `git diff` command failed. Check `agent_output.log` for details.

### Manifold upload issues

**`'manifold' CLI not found; skipping upload.`**

The `manifold` CLI is not installed or not on `$PATH`. Results are still saved locally in `--result_dir`. Install the `manifold` CLI or set `--manifold_base_path ""` to disable uploads.
