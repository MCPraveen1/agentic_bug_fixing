"""LangGraph workflow for repairing a failing Python function.

User code is only executed inside a Docker container. Docker is a useful local
classroom boundary, but this setup is not a hardened multi-tenant sandbox.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

MAX_FIX_ATTEMPTS = 3
DOCKER_IMAGE = os.getenv("BUGFIX_DOCKER_IMAGE", "")
EXECUTION_TIMEOUT_SECONDS = 10
CONTAINER_LABEL = "org.agentic-bug-fixing.execution"
IMAGE_DIGEST_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")

FIX_PROMPT = """You repair a single Python function that failed at runtime.
Return only the complete corrected Python source code, with no markdown fences
or explanation. Preserve the function name, parameters, and intended behavior.
Use the supplied source and error details. Do not add file, network, or process
operations. The code will be run in a constrained local Docker container, but
that is not a guarantee that arbitrary code is safe.
"""


class State(TypedDict, total=False):
    """Data passed between execution, repair, and routing nodes."""

    function_name: str
    source_code: str
    arguments: dict[str, Any] | list[Any]
    error_details: str
    retry_count: int
    execution_result: dict[str, Any]
    original_source_code: str


_CONTAINER_RUNNER = r"""
import json, sys, traceback
payload = json.load(sys.stdin)
namespace = {"__builtins__": __builtins__}
try:
    exec(compile(payload["source_code"], "<submitted-function>", "exec"), namespace, namespace)
    fn = namespace[payload["function_name"]]
    arguments = payload["arguments"]
    result = fn(*arguments) if isinstance(arguments, list) else fn(**arguments)
    try:
        rendered = json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = repr(result)
    print(json.dumps({"ok": True, "result": rendered}))
except BaseException:
    print(json.dumps({"ok": False, "error": traceback.format_exc()}))
    sys.exit(1)
"""


def execute_in_docker(
    source_code: str,
    function_name: str,
    arguments: dict[str, Any] | list[Any],
    *,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one function call in a network-disabled, resource-limited container.

    This deliberately has no local ``exec`` fallback. Docker must be installed
    and available to the current user. Never treat this as a hardened sandbox.
    """
    if not IMAGE_DIGEST_PATTERN.fullmatch(DOCKER_IMAGE):
        return {
            "ok": False,
            "infrastructure_error": True,
            "error": (
                "BUGFIX_DOCKER_IMAGE must be set to an image reference pinned by its "
                "verified sha256 RepoDigest. No floating image tag is allowed."
            ),
        }

    payload = json.dumps(
        {"source_code": source_code, "function_name": function_name, "arguments": arguments}
    )
    execution_id = uuid.uuid4().hex
    container_name = f"agentic-bugfix-{execution_id}"
    command = [
        "docker", "run", "--name", container_name, "--label",
        f"{CONTAINER_LABEL}={execution_id}", "-i", "--network", "none", "--memory", "128m",
        "--cpus", "0.5", "--pids-limit", "64", "--read-only", "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m", "--user", "65534:65534", "--cap-drop",
        "ALL", "--security-opt", "no-new-privileges", DOCKER_IMAGE,
        "python", "-I", "-c", _CONTAINER_RUNNER,
    ]
    timed_out = False
    result: dict[str, Any]
    try:
        completed = subprocess.run(
            command, input=payload, text=True, capture_output=True, timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        result = {
            "ok": False,
            "infrastructure_error": True,
            "error": "Docker CLI is unavailable. Install and start Docker to run submitted code.",
        }
    except subprocess.TimeoutExpired:
        timed_out = True
        result = {"ok": False, "error": f"Execution exceeded the {timeout}-second timeout."}
    except OSError as exc:
        result = {"ok": False, "infrastructure_error": True, "error": f"Docker launch failed: {exc}"}
    else:
        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if output_lines:
            try:
                parsed = json.loads(output_lines[-1])
                if completed.returncode == 0 and parsed.get("ok"):
                    result = parsed
                else:
                    result = {"ok": False, "error": parsed.get("error", completed.stderr.strip())}
            except json.JSONDecodeError:
                result = _docker_command_error(completed)
        else:
            result = _docker_command_error(completed)
    finally:
        cleanup_error = _cleanup_container(container_name, execution_id, terminate=timed_out)
    if cleanup_error:
        result["ok"] = False
        result["cleanup_error"] = cleanup_error
        result["error"] = f"{result.get('error', 'Execution finished.')}; cleanup failed: {cleanup_error}"
    return result


def _docker_command_error(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    message = completed.stderr.strip() or completed.stdout.strip() or "Container execution failed."
    infrastructure_error = any(
        marker in message.lower()
        for marker in (
            "cannot connect to the docker daemon", "error during connect",
            "docker is not running", "is not recognized", "no such image",
            "pull access denied", "conflict. the container name",
        )
    )
    return {"ok": False, "infrastructure_error": infrastructure_error, "error": message}


def _cleanup_container(container_name: str, execution_id: str, *, terminate: bool) -> str | None:
    """Remove only this execution's container, identified by its unique label and ID."""
    inspect_format = f"{{{{.Id}}}}|{{{{index .Config.Labels \"{CONTAINER_LABEL}\"}}}}"
    try:
        inspected = subprocess.run(
            ["docker", "inspect", "--format", inspect_format, container_name],
            text=True, capture_output=True, timeout=5, check=False,
        )
        if inspected.returncode != 0:
            message = inspected.stderr.strip().lower()
            if "no such object" in message or "no such container" in message:
                return None
            return inspected.stderr.strip() or "Could not verify ownership of the execution container."
        container_id, _, label_value = inspected.stdout.strip().partition("|")
        if not container_id or label_value != execution_id:
            return "Container ownership label did not match; cleanup refused to touch it."
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Could not inspect execution container: {exc}"

    if terminate:
        try:
            subprocess.run(
                ["docker", "kill", container_id], text=True, capture_output=True,
                timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            # `rm --force` below is a second termination attempt as well as removal.
            pass
    try:
        removed = subprocess.run(
            ["docker", "rm", "--force", container_id], text=True, capture_output=True,
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Could not remove execution container: {exc}"
    if removed.returncode != 0 and "no such container" not in removed.stderr.lower():
        return removed.stderr.strip() or "Docker did not confirm removal of the execution container."
    return None


def code_execution_node(state: State) -> dict[str, Any]:
    result = execute_in_docker(
        state["source_code"], state["function_name"], state.get("arguments", {})
    )
    error = "" if result.get("ok") else str(result.get("error", "Execution failed."))
    return {"execution_result": result, "error_details": error}


def code_fix_node(state: State) -> dict[str, Any]:
    """Ask ChatOpenAI for a replacement function and count this fixing attempt."""
    attempts = state.get("retry_count", 0) + 1
    prompt = (
        f"Function name: {state['function_name']}\n"
        f"Arguments: {json.dumps(state.get('arguments', {}), ensure_ascii=False)}\n"
        f"Runtime error:\n{state.get('error_details', '')}\n\n"
        f"Current source code:\n{state['source_code']}"
    )
    try:
        model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        response = model.invoke([SystemMessage(content=FIX_PROMPT), HumanMessage(content=prompt)])
        fixed = response.content
        if not isinstance(fixed, str):
            raise ValueError("The model returned a non-text response.")
        fixed = re.sub(r"^\s*```(?:python)?\s*|\s*```\s*$", "", fixed.strip(), flags=re.IGNORECASE)
        if not fixed.strip():
            raise ValueError("The model returned empty source code.")
        return {"source_code": fixed, "retry_count": attempts, "error_details": ""}
    except Exception as exc:
        return {
            "retry_count": attempts,
            "error_details": f"OpenAI fix attempt failed: {type(exc).__name__}: {exc}",
        }


def error_router(state: State) -> str:
    """Stop on success or infrastructure failure; otherwise retry up to 3 fixes."""
    result = state.get("execution_result", {})
    if result.get("ok") or result.get("infrastructure_error"):
        return END
    if state.get("retry_count", 0) >= MAX_FIX_ATTEMPTS:
        return END
    return "code_fix_node"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("code_execution_node", code_execution_node)
    graph.add_node("code_fix_node", code_fix_node)
    graph.set_entry_point("code_execution_node")
    graph.add_conditional_edges(
        "code_execution_node", error_router,
        {"code_fix_node": "code_fix_node", END: END},
    )
    graph.add_edge("code_fix_node", "code_execution_node")
    return graph.compile()


agent_graph = build_graph()


def run_agent(
    source_code: str,
    function_name: str,
    arguments: dict[str, Any] | list[Any] | None = None,
) -> State:
    """Execute and, when needed, repair a function with at most three fixes."""
    initial: State = {
        "function_name": function_name,
        "source_code": source_code,
        "original_source_code": source_code,
        "arguments": arguments or {},
        "error_details": "",
        "retry_count": 0,
        "execution_result": {},
    }
    return agent_graph.invoke(initial, config={"recursion_limit": 12})
