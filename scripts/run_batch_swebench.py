#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Batch evaluation script for running Confucius Code Agent on SWE-Bench Pro problems.

This script automates running Confucius on multiple SWE-Bench Pro problems, managing
the podman container lifecycle (start → inject prompt → run agent → extract results
→ cleanup), A/B experiment configurations, AWS credential refresh,
dry-run mode, and producing a summary report.

Usage:
    python scripts/run_batch_swebench.py [options]

See SWEBENCH_README.md for detailed documentation.
"""

import argparse
import base64
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(result_dir: Path, log_filename: str = "batch_run.log") -> logging.Logger:
    """Set up logging to both stdout and a file inside result_dir."""
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / log_filename

    logger = logging.getLogger("run_batch_swebench")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

CONFIG_NO_COMPRESSION = "no_compression"
CONFIG_SMART_CONTEXT = "smart_context"
ALL_CONFIGS = [CONFIG_NO_COMPRESSION, CONFIG_SMART_CONTEXT]

# Files that are batch-level metadata and should not be uploaded to GitHub
_GITHUB_UPLOAD_SKIP_FILES = {"batch_run.log", "summary.json"}


def config_dir_name(config: str, args: argparse.Namespace) -> str:
    """Return the directory name for a given experiment config, encoding parameters.

    For ``smart_context`` the directory name includes the compression threshold and
    clear-at-least values so that runs with different parameters do not overwrite
    each other, e.g. ``smart_context_50000_20000``.
    """
    if config == CONFIG_SMART_CONTEXT:
        return f"smart_context_{args.compression_threshold}_{args.clear_at_least}"
    return config


# Proxy settings passed into every container and podman exec call
_NO_PROXY = "localhost,127.0.0.1,*.facebook.com,*.tfbnw.net,*.fb.com,*.internalfb.com"
_PROXY_URL = "http://127.0.0.1:18080"


def build_agent_flags(
    config: str,
    compression_threshold: int,
    clear_at_least: int,
    clear_at_least_tolerance: float,
    enable_context_usage: bool = False,
    context_window_size: Optional[int] = None,
    enable_compression_agent: bool = False,
    compression_agent_model: Optional[str] = None,
    compression_agent_max_tokens: Optional[int] = None,
) -> list[str]:
    """Return the CLI flags to pass to the agent for the given experiment config."""
    if config == CONFIG_SMART_CONTEXT:
        flags = [
            "--cache-min-prompt-length", "0",
            "--verbose",
            "--enable-smart-context",
            "--disable-reminder",
            "--compression-threshold", str(compression_threshold),
            "--clear-at-least", str(clear_at_least),
            "--clear-at-least-tolerance", str(clear_at_least_tolerance),
            "--context-edit-log-dir", "/app",
        ]
        if enable_context_usage:
            flags.append("--enable-context-usage")
            if context_window_size is not None:
                flags.extend(["--context-window-size", str(context_window_size)])
        if enable_compression_agent:
            flags.append("--enable-compression-agent")
            if compression_agent_model is not None:
                flags.extend(["--compression-agent-model", compression_agent_model])
            if compression_agent_max_tokens is not None:
                flags.extend(["--compression-agent-max-tokens", str(compression_agent_max_tokens)])
        return flags
    # no_compression — minimal flags
    return ["--cache-min-prompt-length", "0"]


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def load_problems(problems_file: Path, problem_ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """
    Load problems from the problems file.

    The file may be a JSON array or newline-delimited JSON (JSONL).  Each entry
    must contain at least ``instance_id`` and ``problem_statement``.
    """
    text = problems_file.read_text(encoding="utf-8").strip()

    # Detect whether it's a JSON array or JSONL
    if text.startswith("["):
        problems: list[dict[str, Any]] = json.loads(text)
    else:
        problems = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    if problem_ids:
        id_set = set(problem_ids)
        problems = [p for p in problems if p["instance_id"] in id_set]

    return problems


# ---------------------------------------------------------------------------
# Image name derivation
# ---------------------------------------------------------------------------

def derive_image_name(problem: dict[str, Any], image_registry: str) -> str:
    """
    Derive the podman image name from a problem record.

    Uses ``{image_registry}/{instance_id}`` where instance_id comes from the
    JSONL record.  If the problem record already contains an ``image_name``
    field it is used as-is (for override/compatibility).
    """
    if "image_name" in problem and problem["image_name"]:
        return problem["image_name"]

    instance_id: str = problem["instance_id"]
    return f"{image_registry}/{instance_id}"


# ---------------------------------------------------------------------------
# Container lifecycle helpers
# ---------------------------------------------------------------------------

def container_name(instance_id: str, config: str) -> str:
    """Return a deterministic, filesystem-safe container name."""
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", instance_id)
    return f"cf_sweb_{safe_id}_{config}"


def run_cmd(
    cmd: list[str],
    logger: logging.Logger,
    timeout: Optional[int] = None,
    check: bool = True,
    capture_output: bool = True,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with logging."""
    logger.debug("Running command: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
        check=check,
        env=env,
    )
    if result.stdout:
        logger.debug("stdout: %s", result.stdout[:2000])
    if result.stderr:
        logger.debug("stderr: %s", result.stderr[:2000])
    return result


def start_container(
    runtime: str,
    image: str,
    name: str,
    instance_id: str,
    problem_statement: str,
    data_dir: str,
    container_memory: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> bool:
    """Start a detached container and return True on success."""
    # Build the base podman run command (wrapped with with-proxy for network access)
    cmd = [
        "with-proxy", runtime, "run",
        "--name", name,
        "-d",
        "--rm",
        "--network", "host",
        f"--memory={container_memory}",
        f"--memory-swap={container_memory}",
        "--userns=host",
        "-v", f"{data_dir}:/data",
        "--entrypoint", "/bin/bash",
        # Task identifiers
        "-e", f"TASK_ID={instance_id}",
        "-e", f"PROBLEM_STATEMENT={problem_statement}",
        # AWS region
        "-e", "AWS_REGION_NAME=us-west-2",
        "-e", "AWS_DEFAULT_REGION=us-west-2",
        "-e", "AWS_REGION=us-west-2",
        # AWS credentials (pass through from host env)
        "-e", "AWS_ACCESS_KEY_ID",
        "-e", "AWS_SECRET_ACCESS_KEY",
        "-e", "AWS_SESSION_TOKEN",
        "-e", "EXPIRY_TIME",
        "-e", "SESSION_DURATION_SECONDS",
        # Proxy settings
        "-e", f"http_proxy={_PROXY_URL}",
        "-e", f"https_proxy={_PROXY_URL}",
        "-e", f"ftp_proxy={_PROXY_URL}",
        "-e", f"no_proxy={_NO_PROXY}",
        image,
        "-c", "sleep infinity",
    ]
    if dry_run:
        logger.info("[dry-run] Would run: %s", " ".join(shlex.quote(a) for a in cmd))
        return True
    try:
        run_cmd(cmd, logger, timeout=120)
        logger.info("Started container: %s (image=%s)", name, image)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to start container %s: %s\n%s", name, e, e.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Timed out starting container %s", name)
        return False


def setup_container_environment(
    runtime: str,
    name: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> bool:
    """Bootstrap the conda environment and copy the PEX inside the container."""
    setup_script = (
        "export AWS_REGION_NAME=us-west-2 && "
        "export AWS_DEFAULT_REGION=us-west-2 && "
        "export AWS_REGION=us-west-2 && "
        "mkdir -p /opt/appenv && "
        "tar --no-same-owner -xzf /data/cf_env.tar.gz -C /opt/appenv && "
        "/opt/appenv/bin/conda-unpack && "
        "cp /data/app.pex /usr/local/bin/"
    )
    cmd = [runtime, "exec", name, "/bin/bash", "-c", setup_script]

    if dry_run:
        logger.info("[dry-run] Would set up container environment in %s", name)
        return True

    logger.info("Setting up environment in container %s…", name)
    try:
        run_cmd(cmd, logger, timeout=300)
        logger.info("Container environment ready: %s", name)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to set up container environment %s: %s\n%s", name, e, e.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Timed out setting up container environment %s", name)
        return False


def stop_container(
    runtime: str,
    name: str,
    logger: logging.Logger,
) -> None:
    """Stop and remove a container, ignoring errors."""
    try:
        run_cmd([runtime, "stop", name], logger, timeout=30, check=False)
    except Exception as e:
        logger.debug("Error stopping container %s: %s", name, e)
    try:
        run_cmd([runtime, "rm", "-f", name], logger, timeout=30, check=False)
    except Exception as e:
        logger.debug("Error removing container %s: %s", name, e)


def write_prompt_to_container(
    runtime: str,
    name: str,
    problem_statement: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> bool:
    """Write the problem statement to /tmp/task.txt inside the container."""
    # Build the full SWE-Bench prompt (same template as run_swebench.py)
    prompt = _build_swebench_prompt(problem_statement)

    if dry_run:
        logger.info("[dry-run] Would write prompt (%d chars) to %s:/tmp/task.txt", len(prompt), name)
        return True

    # Pass the prompt via stdin to avoid shell quoting issues with arbitrary content
    try:
        proc = subprocess.run(
            [runtime, "exec", "-i", name, "sh", "-c", "cat > /tmp/task.txt"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            logger.error("Failed to write prompt to container %s: %s", name, proc.stderr)
            return False
        logger.debug("Wrote prompt (%d chars) to %s:/tmp/task.txt", len(prompt), name)
        return True
    except subprocess.TimeoutExpired:
        logger.error("Timed out writing prompt to container %s", name)
        return False


def _build_swebench_prompt(problem_statement: str) -> str:
    """Build the full SWE-Bench prompt matching the template in run_swebench.py."""
    # Use the same template as the single-task runner (with safe substitution)
    escaped = problem_statement.replace("$", "$$")
    from string import Template
    template = Template(
        """## Work directory
I've uploaded a python code repository in your current directory, this will be the repository for you to investigate and make code changes.

## Problem Statement
$problem_statement

## Your Task
Can you help me implement the necessary changes to the repository so that the requirements specified in the problem statement are met?
I've already taken care of all changes to any of the test files described in the problem statement. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the $${working_dir} directory to ensure the problem statement is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the problem statement
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
3. Edit the source code of the repo to resolve the issue
4. Rerun your reproduction script and confirm that the error is fixed!
5. Think about edge cases and make sure your fix handles them as well

**Note**: this is a HARD problem, which means you need to think HARD! Your thinking should be thorough and so it's fine if it's very long.
**Note**: you are not allowed to modify project dependency files like `pyproject.toml` or `setup.py` or `requirements.txt` or `package.json`

## Exit Criteria
Please carefully follow the steps below to help review your changes.
    1. If you made any changes to your code after running the reproduction script, please run the reproduction script again.
    If the reproduction script is failing, please revisit your changes and make sure they are correct.
    If you have already removed your reproduction script, please ignore this step.

    2. Remove your reproduction script (if you haven't done so already).

    3. If you have modified any TEST files, please revert them to the state they had before you started fixing the issue.
    You can do this with `git checkout -- /path/to/test/file.py`. Use below <diff> to find the files you need to revert.

    4. Commit your change, make sure you only have one commit.
Plz make sure you commit your change at the end, otherwise I won't be able to export your change.
"""
    )
    return template.substitute(problem_statement=escaped)


def run_agent_in_container(
    runtime: str,
    name: str,
    pex_path: str,
    agent_flags: list[str],
    logger: logging.Logger,
    task_timeout: int = 3600,
    dry_run: bool = False,
) -> tuple[int, str]:
    """
    Run the Confucius agent inside the container.

    Returns (exit_code, combined_output).
    """
    agent_cmd = [
        runtime, "exec",
        "-w", "/app",
        "-e", "PEX_PYTHON=/opt/appenv/bin/python",
        "-e", "AWS_REGION_NAME=us-west-2",
        "-e", "AWS_DEFAULT_REGION=us-west-2",
        "-e", "AWS_REGION=us-west-2",
    ]
    # Forward all credential and proxy env vars
    for var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "EXPIRY_TIME",
        "SESSION_DURATION_SECONDS",
    ):
        value = os.environ.get(var)
        if value:
            agent_cmd += ["-e", f"{var}={value}"]
    # Proxy env vars
    agent_cmd += [
        "-e", f"http_proxy={_PROXY_URL}",
        "-e", f"https_proxy={_PROXY_URL}",
        "-e", f"ftp_proxy={_PROXY_URL}",
        "-e", f"no_proxy={_NO_PROXY}",
    ]
    agent_cmd += [
        name,
        "python", pex_path,
        "--prompt", "/tmp/task.txt",
        "--verbose",
    ] + agent_flags

    if dry_run:
        logger.info("[dry-run] Would run agent: %s", " ".join(shlex.quote(a) for a in agent_cmd))
        return 0, "[dry-run] agent output"

    logger.info("Running agent in container %s (timeout=%ds)…", name, task_timeout)
    logger.debug("Agent command: %s", " ".join(agent_cmd))

    try:
        proc = subprocess.run(
            agent_cmd,
            capture_output=True,
            text=True,
            timeout=task_timeout,
        )
        combined = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
        logger.info(
            "Agent finished in container %s (exit_code=%d)", name, proc.returncode
        )
        return proc.returncode, combined
    except subprocess.TimeoutExpired:
        logger.warning("Agent timed out in container %s after %ds", name, task_timeout)
        return -1, f"TIMEOUT after {task_timeout}s"
    except Exception as e:
        logger.error("Error running agent in container %s: %s", name, e)
        return -1, str(e)


def extract_results(
    runtime: str,
    name: str,
    out_dir: Path,
    logger: logging.Logger,
    dry_run: bool = False,
) -> None:
    """
    Extract all result artifacts from the container to out_dir.

    Artifacts:
      - patch.diff         — committed changes via ``git show``
      - logs.txt           — agent log file at /app/logs.txt
      - context_edits.jsonl — context edit log at /app/context_edits.jsonl (smart_context only)
      - traj_*.json        — trajectory files at /tmp/confucius/traj_*.json
      - token_usage.json   — token usage at /app/token_usage.json
    """
    if dry_run:
        logger.info("[dry-run] Would extract results from container %s to %s", name, out_dir)
        return

    # 1. patch.diff — use git show to get the committed patch
    try:
        result = run_cmd(
            [runtime, "exec", name, "git", "show", "--format=", "--no-color"],
            logger,
            timeout=60,
        )
        (out_dir / "patch.diff").write_text(result.stdout, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to extract patch.diff from %s: %s", name, e.stderr)
        (out_dir / "patch.diff").write_text(f"ERROR: {e.stderr}", encoding="utf-8")
    except subprocess.TimeoutExpired:
        logger.warning("Timed out extracting patch.diff from %s", name)
        (out_dir / "patch.diff").write_text("ERROR: timeout", encoding="utf-8")

    # Helper to copy a single file out of the container
    def _podman_cp(container_path: str, local_name: str) -> None:
        try:
            run_cmd(
                [runtime, "cp", f"{name}:{container_path}", str(out_dir / local_name)],
                logger,
                timeout=60,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("Could not copy %s from %s: %s", container_path, name, e.stderr)
        except subprocess.TimeoutExpired:
            logger.warning("Timed out copying %s from %s", container_path, name)

    # 2. logs.txt
    _podman_cp("/app/logs.txt", "logs.txt")

    # 2b. context_edits.jsonl — context edit log (if smart context was used)
    _podman_cp("/app/context_edits.jsonl", "context_edits.jsonl")

    # 3. traj_*.json — list matching files inside the container then copy each
    try:
        ls_result = run_cmd(
            [runtime, "exec", name, "sh", "-c", "ls /tmp/confucius/traj_*.json 2>/dev/null || true"],
            logger,
            timeout=30,
        )
        for path in ls_result.stdout.splitlines():
            path = path.strip()
            if path:
                filename = Path(path).name
                _podman_cp(path, filename)
    except Exception as e:
        logger.warning("Could not list trajectory files in %s: %s", name, e)

    # 4. token_usage.json
    _podman_cp("/app/token_usage.json", "token_usage.json")


# ---------------------------------------------------------------------------
# AWS credential refresh
# ---------------------------------------------------------------------------

def refresh_aws_credentials(logger: logging.Logger) -> bool:
    """Refresh AWS credentials using `cloud aws get-creds` and apply them to os.environ."""
    logger.info("Refreshing AWS credentials…")
    try:
        result = subprocess.run(
            ["cloud", "aws", "get-creds", "009160068926", "--role", "SSOAdmin", "--duration", "14400"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            # Parse `export VAR=value` lines and apply to the current process env.
            # Use shlex.split to correctly handle quoted values (including values
            # that may contain spaces or special characters).
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line.startswith("export "):
                    continue
                try:
                    parts = shlex.split(line[len("export "):])
                    for part in parts:
                        if "=" in part:
                            key, _, value = part.partition("=")
                            os.environ[key] = value
                            logger.debug("Set env var: %s", key)
                except ValueError:
                    logger.debug("Could not parse credential line: %s", line)
            logger.info("AWS credentials refreshed successfully.")
            return True
        else:
            logger.warning(
                "AWS credential refresh returned exit code %d: %s",
                result.returncode,
                result.stderr,
            )
            return False
    except FileNotFoundError:
        logger.warning("'cloud' CLI not found; skipping credential refresh.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("AWS credential refresh timed out.")
        return False


# ---------------------------------------------------------------------------
# Per-problem runner
# ---------------------------------------------------------------------------

def run_problem(
    problem: dict[str, Any],
    config: str,
    args: argparse.Namespace,
    result_base_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Run the agent on a single problem with the given experiment configuration.

    Returns a metadata dict with fields: instance_id, config, success, exit_code,
    duration_seconds, image, error.
    """
    instance_id: str = problem["instance_id"]
    problem_statement: str = problem["problem_statement"]
    image = derive_image_name(problem, args.image_registry)
    dir_name = config_dir_name(config, args)
    cname = container_name(instance_id, dir_name)

    out_dir = result_base_dir / dir_name / instance_id
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_flags = build_agent_flags(
        config,
        args.compression_threshold,
        args.clear_at_least,
        args.clear_at_least_tolerance,
        args.enable_context_usage,
        args.context_window_size,
        args.enable_compression_agent,
        args.compression_agent_model,
        args.compression_agent_max_tokens,
    )

    logger.info(
        "─── Starting problem: %s | config: %s | image: %s ───",
        instance_id,
        config,
        image,
    )

    metadata: dict[str, Any] = {
        "instance_id": instance_id,
        "config": config,
        "image": image,
        "container_name": cname,
        "started_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "success": False,
        "exit_code": None,
        "duration_seconds": None,
        "error": None,
    }

    t_start = time.monotonic()

    # Retry loop for the entire problem attempt
    for attempt in range(1, args.max_retries + 1):
        if attempt > 1:
            logger.info("Retrying %s/%s (attempt %d/%d)…", instance_id, config, attempt, args.max_retries)

        try:
            # 1. Start container
            started = start_container(
                args.container_runtime,
                image,
                cname,
                instance_id,
                problem_statement,
                args.data_dir,
                args.container_memory,
                logger,
                args.dry_run,
            )
            if not started:
                raise RuntimeError(f"Could not start container {cname}")

            # 2. Set up container environment (conda unpack + copy PEX)
            setup_ok = setup_container_environment(
                args.container_runtime, cname, logger, args.dry_run
            )
            if not setup_ok:
                raise RuntimeError(f"Could not set up environment in container {cname}")

            # 3. Write prompt
            wrote = write_prompt_to_container(
                args.container_runtime, cname, problem_statement, logger, args.dry_run
            )
            if not wrote:
                raise RuntimeError(f"Could not write prompt to container {cname}")

            # 4. Run agent
            exit_code, agent_output = run_agent_in_container(
                args.container_runtime,
                cname,
                args.pex_path,
                agent_flags,
                logger,
                task_timeout=args.task_timeout,
                dry_run=args.dry_run,
            )

            # 5. Extract all result artifacts
            extract_results(args.container_runtime, cname, out_dir, logger, args.dry_run)

            # 6. Write agent output log and metadata
            (out_dir / "agent_output.log").write_text(agent_output, encoding="utf-8")

            duration = time.monotonic() - t_start
            metadata.update(
                {
                    "success": exit_code == 0,
                    "exit_code": exit_code,
                    "duration_seconds": round(duration, 1),
                    "finished_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                }
            )
            (out_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            logger.info(
                "Completed %s/%s in %.1fs (exit_code=%d)",
                instance_id,
                config,
                duration,
                exit_code,
            )
            break  # success — exit retry loop

        except Exception as e:
            logger.warning(
                "Attempt %d/%d failed for %s/%s: %s",
                attempt,
                args.max_retries,
                instance_id,
                config,
                e,
            )
            metadata["error"] = str(e)
            if attempt == args.max_retries:
                duration = time.monotonic() - t_start
                metadata["duration_seconds"] = round(duration, 1)
                metadata["finished_at"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                (out_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
        finally:
            # Always stop/remove container
            if not args.dry_run:
                stop_container(args.container_runtime, cname, logger)

    return metadata


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(
    results: list[dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """Print a human-readable summary of all results."""
    total = len(results)
    succeeded = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total:           %d", total)
    logger.info("Agent succeeded: %d", len(succeeded))
    logger.info("Agent failed:    %d", len(failed))

    if failed:
        logger.info("")
        logger.info("Failed agent runs:")
        for r in failed:
            logger.info(
                "  • %s [%s] exit_code=%s error=%s",
                r["instance_id"],
                r["config"],
                r.get("exit_code"),
                r.get("error"),
            )

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# GitHub upload
# ---------------------------------------------------------------------------

_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB — GitHub Contents API limit


def upload_results_to_github(
    result_dir: Path,
    github_results_repo: str,
    github_results_path: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> None:
    """Upload all files in result_dir to a GitHub repository using the Contents API.

    Files are written to ``{github_results_path}/...`` in the target
    repo, mirroring the local directory structure under ``result_dir``.

    Requires the ``GITHUB_TOKEN`` environment variable to be set with a fine-grained
    PAT that has Contents read/write permission on the target repository.

    Requests are routed through the proxy at ``_PROXY_URL`` as required on the devvm.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning(
            "GITHUB_TOKEN is not set; skipping GitHub upload. "
            "Set GITHUB_TOKEN with Contents read/write permission on %s.",
            github_results_repo,
        )
        return

    proxy_handler = urllib.request.ProxyHandler(
        {"http": _PROXY_URL, "https": _PROXY_URL}
    )
    opener = urllib.request.build_opener(proxy_handler)

    api_base = f"{_GITHUB_API_BASE}/repos/{github_results_repo}/contents"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    files = sorted(p for p in result_dir.rglob("*") if p.is_file())
    logger.info(
        "Uploading %d file(s) to %s/%s/",
        len(files),
        github_results_repo,
        github_results_path,
    )

    for file_path in files:
        rel_path = file_path.relative_to(result_dir)

        # Skip batch-level metadata files that live at the root of result_dir
        if len(rel_path.parts) == 1 and rel_path.name in _GITHUB_UPLOAD_SKIP_FILES:
            logger.info("Skipping local-only file: %s", rel_path)
            continue

        # Reorder path from config/instance_id/... to instance_id/config/... so that
        # results are grouped by task first and then by policy in the target repo.
        if len(rel_path.parts) >= 2:
            upload_rel = Path(rel_path.parts[1], rel_path.parts[0], *rel_path.parts[2:])
        else:
            # Defensive fallback for unexpected single-component paths
            upload_rel = rel_path
        github_path = f"{github_results_path}/{upload_rel}"

        file_size = file_path.stat().st_size
        if file_size > _GITHUB_MAX_FILE_BYTES:
            logger.warning(
                "Skipping %s — file too large (%d bytes > 50 MB limit)",
                rel_path,
                file_size,
            )
            continue

        if dry_run:
            logger.info("[dry-run] Would upload %s → %s:%s", rel_path, github_results_repo, github_path)
            continue

        content_b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
        body = json.dumps(
            {
                "message": f"Add SBP eval result: {rel_path}",
                "content": content_b64,
            }
        ).encode("utf-8")

        url = f"{api_base}/{github_path}"
        req = urllib.request.Request(url, data=body, headers=headers, method="PUT")
        try:
            with opener.open(req) as resp:
                status = resp.status
            logger.info("Uploaded %s (HTTP %d)", github_path, status)
        except urllib.error.HTTPError as exc:
            logger.error("Failed to upload %s: HTTP %d %s", github_path, exc.code, exc.reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to upload %s: %s", github_path, exc)


def upload_task_results_to_github(
    out_dir: Path,
    dir_name: str,
    instance_id: str,
    github_results_repo: str,
    github_results_path: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> None:
    """Upload a single task's result directory to GitHub immediately after it completes.

    Files under ``out_dir`` (which is ``result_dir/dir_name/instance_id/``) are uploaded
    to ``{github_results_path}/{instance_id}/{dir_name}/`` in the target repo, so that
    results are grouped by task first and then by policy.

    Requires the ``GITHUB_TOKEN`` environment variable to be set with a fine-grained
    PAT that has Contents read/write permission on the target repository.

    Requests are routed through the proxy at ``_PROXY_URL`` as required on the devvm.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning(
            "GITHUB_TOKEN is not set; skipping GitHub upload. "
            "Set GITHUB_TOKEN with Contents read/write permission on %s.",
            github_results_repo,
        )
        return

    proxy_handler = urllib.request.ProxyHandler(
        {"http": _PROXY_URL, "https": _PROXY_URL}
    )
    opener = urllib.request.build_opener(proxy_handler)

    api_base = f"{_GITHUB_API_BASE}/repos/{github_results_repo}/contents"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    logger.info(
        "Uploading %d file(s) for %s/%s to %s/%s/",
        len(files),
        dir_name,
        instance_id,
        github_results_repo,
        github_results_path,
    )

    for file_path in files:
        rel_path = file_path.relative_to(out_dir)
        # Place results at instance_id/dir_name/... so tasks are grouped by instance first
        github_path = f"{github_results_path}/{instance_id}/{dir_name}/{rel_path}"

        file_size = file_path.stat().st_size
        if file_size > _GITHUB_MAX_FILE_BYTES:
            logger.warning(
                "Skipping %s — file too large (%d bytes > 50 MB limit)",
                rel_path,
                file_size,
            )
            continue

        if dry_run:
            logger.info("[dry-run] Would upload %s → %s:%s", rel_path, github_results_repo, github_path)
            continue

        content_b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
        body = json.dumps(
            {
                "message": f"Add SBP eval result: {instance_id}/{dir_name}/{rel_path}",
                "content": content_b64,
            }
        ).encode("utf-8")

        url = f"{api_base}/{github_path}"
        req = urllib.request.Request(url, data=body, headers=headers, method="PUT")
        try:
            with opener.open(req) as resp:
                status = resp.status
            logger.info("Uploaded %s (HTTP %d)", github_path, status)
        except urllib.error.HTTPError as exc:
            logger.error("Failed to upload %s: HTTP %d %s", github_path, exc.code, exc.reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to upload %s: %s", github_path, exc)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluation script for running Confucius Code Agent on "
            "SWE-Bench Pro problems using podman containers."
        )
    )

    # Problem selection
    parser.add_argument(
        "--problems_file",
        type=Path,
        default=Path("sbp-problems.jsonl"),
        help="Path to the problems file (JSON array or JSONL). Default: sbp-problems.jsonl",
    )
    parser.add_argument(
        "--problem_ids",
        nargs="+",
        metavar="INSTANCE_ID",
        default=None,
        help="Run only the specified instance IDs (space-separated).",
    )

    # Experiment configurations
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=ALL_CONFIGS,
        default=ALL_CONFIGS,
        help=(
            "Experiment configurations to run. "
            f"Choices: {ALL_CONFIGS}. Default: both."
        ),
    )

    # Smart context thresholds
    parser.add_argument(
        "--compression_threshold",
        type=int,
        default=50000,
        help="Token count triggering context compression (smart_context config). Default: 50000",
    )
    parser.add_argument(
        "--clear_at_least",
        type=int,
        default=20000,
        help="Minimum tokens cleared per compression operation. Default: 20000",
    )
    parser.add_argument(
        "--clear_at_least_tolerance",
        type=float,
        default=0.75,
        help="Tolerance for clear-at-least compression. Default: 0.75",
    )
    parser.add_argument(
        "--enable_context_usage",
        action="store_true",
        default=False,
        help="Include cumulative context usage in system_info tags (smart_context config only).",
    )
    parser.add_argument(
        "--context_window_size",
        type=int,
        default=None,
        help="Total context window size in tokens for usage percentage. Default: 200000",
    )

    # Compression agent
    parser.add_argument(
        "--enable_compression_agent",
        action="store_true",
        default=False,
        help="Enable the two-agent compression architecture. "
        "Main agent provides guidance, compression agent handles rewriting.",
    )
    parser.add_argument(
        "--compression_agent_model",
        type=str,
        default=None,
        help="Model ID for the compression agent (default: same as main agent).",
    )
    parser.add_argument(
        "--compression_agent_max_tokens",
        type=int,
        default=None,
        help="Max tokens for compression agent responses (default 16384).",
    )

    # Container / agent
    parser.add_argument(
        "--pex_path",
        type=str,
        default="/usr/local/bin/app.pex",
        help="Path to the app.pex binary inside the container. Default: /usr/local/bin/app.pex",
    )
    parser.add_argument(
        "--container_runtime",
        type=str,
        default="podman",
        choices=["podman", "docker"],
        help="Container runtime to use. Default: podman",
    )
    parser.add_argument(
        "--image_registry",
        type=str,
        default="vmvm-registry.fbinfra.net/sweap_retag",
        help=(
            "Container image registry prefix. The instance_id from the JSONL is appended "
            "to form the full image name. Default: vmvm-registry.fbinfra.net/sweap_retag"
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/workspace/sbp_memory"),
        help=(
            "Host path mounted as /data inside the container (supplies cf_env.tar.gz and app.pex). "
            "Default: ~/workspace/sbp_memory"
        ),
    )
    parser.add_argument(
        "--container_memory",
        type=str,
        default="4g",
        help="Memory limit for each container (passed to --memory and --memory-swap). Default: 4g",
    )

    # Results
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=Path("results_swebench"),
        help="Directory to store results. Default: ./results_swebench",
    )

    # Timing / retry
    parser.add_argument(
        "--task_timeout",
        type=int,
        default=3600,
        help="Per-problem timeout in seconds. Default: 3600",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts per problem. Default: 3",
    )

    # AWS credentials
    parser.add_argument(
        "--skip_credential_refresh",
        action="store_true",
        default=False,
        help="Skip AWS credential refresh before the run.",
    )
    parser.add_argument(
        "--credential_refresh_interval",
        type=int,
        default=0,
        metavar="SECONDS",
        help=(
            "Refresh AWS credentials every N seconds during the run "
            "(0 = only refresh once at the start). Default: 0"
        ),
    )

    # Misc
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Dry-run mode: print commands without executing them.",
    )

    # GitHub upload
    parser.add_argument(
        "--github_results_repo",
        type=str,
        default="samuellin01/memory_experiments",
        help=(
            "Target GitHub repository (owner/name) to upload results to. "
            "Default: samuellin01/memory_experiments"
        ),
    )
    parser.add_argument(
        "--github_results_path",
        type=str,
        default="sbp",
        help=(
            "Base path/folder inside the target GitHub repo where results are written. "
            "Results land at {github_results_path}/{timestamp}/... "
            "Default: sbp"
        ),
    )
    parser.add_argument(
        "--skip_github_upload",
        action="store_true",
        default=False,
        help="Skip uploading results to GitHub after the run.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Stamp the run directory with a timestamp so repeated runs don't collide
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir: Path = args.result_dir / timestamp
    logger = setup_logging(result_dir)

    logger.info("SWE-Bench Pro Batch Evaluation")
    logger.info("  problems_file:           %s", args.problems_file)
    logger.info("  configs:                 %s", args.configs)
    logger.info("  result_dir:              %s", result_dir)
    logger.info("  container_runtime:       %s", args.container_runtime)
    logger.info("  image_registry:          %s", args.image_registry)
    logger.info("  data_dir:                %s", args.data_dir)
    logger.info("  container_memory:        %s", args.container_memory)
    logger.info("  pex_path:                %s", args.pex_path)
    logger.info("  task_timeout:            %ds", args.task_timeout)
    logger.info("  max_retries:             %d", args.max_retries)
    logger.info("  dry_run:                 %s", args.dry_run)
    logger.info("  skip_github_upload:      %s", args.skip_github_upload)
    if not args.skip_github_upload:
        logger.info("  github_results_repo:     %s", args.github_results_repo)
        logger.info("  github_results_path:     %s", args.github_results_path)
    if args.problem_ids:
        logger.info("  problem_ids:             %s", args.problem_ids)
    if CONFIG_SMART_CONTEXT in args.configs:
        logger.info("  compression_threshold:   %d", args.compression_threshold)
        logger.info("  clear_at_least:          %d", args.clear_at_least)
        logger.info("  clear_at_least_tolerance: %s", args.clear_at_least_tolerance)
        logger.info("  enable_context_usage:    %s", args.enable_context_usage)
        if args.enable_context_usage and args.context_window_size is not None:
            logger.info("  context_window_size:     %d", args.context_window_size)
        if args.enable_compression_agent:
            logger.info("  compression_agent:       ENABLED")
            if args.compression_agent_model is not None:
                logger.info("  compression_agent_model: %s", args.compression_agent_model)
            if args.compression_agent_max_tokens is not None:
                logger.info("  compression_agent_max_tokens: %d", args.compression_agent_max_tokens)

    # Load problems
    if not args.problems_file.exists():
        logger.error("Problems file not found: %s", args.problems_file)
        sys.exit(1)

    problems = load_problems(args.problems_file, args.problem_ids)
    if not problems:
        logger.error("No problems found (check --problem_ids filter and problems file).")
        sys.exit(1)

    logger.info("Loaded %d problem(s).", len(problems))

    # AWS credentials
    if not args.skip_credential_refresh:
        refresh_aws_credentials(logger)

    last_credential_refresh = time.monotonic()

    # Main run loop
    all_results: list[dict[str, Any]] = []

    for problem_idx, problem in enumerate(problems, start=1):
        instance_id = problem["instance_id"]

        # Refresh credentials before each problem (tasks can take up to 3600s)
        if not args.skip_credential_refresh:
            refresh_aws_credentials(logger)
            last_credential_refresh = time.monotonic()

        for config in args.configs:
            logger.info(
                "[%d/%d] Problem: %s | Config: %s",
                problem_idx,
                len(problems),
                instance_id,
                config,
            )
            meta = run_problem(problem, config, args, result_dir, logger)
            all_results.append(meta)

            dir_name = config_dir_name(config, args)
            out_dir = result_dir / dir_name / instance_id

            # Upload this task's results immediately so partial results are preserved
            # even if the batch is interrupted.
            if not args.skip_github_upload:
                if meta.get("success"):
                    upload_task_results_to_github(
                        out_dir=out_dir,
                        dir_name=dir_name,
                        instance_id=instance_id,
                        github_results_repo=args.github_results_repo,
                        github_results_path=args.github_results_path,
                        logger=logger,
                        dry_run=args.dry_run,
                    )
                else:
                    logger.info(
                        "Skipping GitHub upload for %s/%s — agent execution failed (exit_code=%s)",
                        instance_id,
                        dir_name,
                        meta.get("exit_code"),
                    )

    # Save full results summary
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info("Full results written to %s", summary_path)

    # Print human-readable summary
    print_summary(all_results, logger)

    # Exit with non-zero if any problem failed
    failed_count = sum(1 for r in all_results if not r.get("success"))
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
