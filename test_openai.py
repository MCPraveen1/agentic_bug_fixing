"""Basic workflow tests; all model and container interactions are mocked."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import agentic_bug_fixing_openai as app


TEST_IMAGE = "python:3.12-slim@sha256:" + "a" * 64


def _docker_responses(run_result):
    def respond(command, **_kwargs):
        if command[1] == "run":
            return run_result
        if command[1] == "inspect":
            label = command[-1].removeprefix("agentic-bugfix-")
            return subprocess.CompletedProcess(command, 0, f"container-id|{label}\n", "")
        if command[1] == "kill":
            return subprocess.CompletedProcess(command, 0, "container-id\n", "")
        if command[1] == "rm":
            return subprocess.CompletedProcess(command, 0, "container-id\n", "")
        raise AssertionError(f"Unexpected Docker command: {command}")

    return respond


def test_successful_execution():
    with patch.object(app, "execute_in_docker", return_value={"ok": True, "result": "5"}):
        result = app.run_agent("def add(a, b): return a + b", "add", {"a": 2, "b": 3})
    assert result["execution_result"]["ok"] is True
    assert result["retry_count"] == 0
    assert result["original_source_code"] == result["source_code"]


def test_runtime_error_is_fixed_and_retried():
    executions = iter([
        {"ok": False, "error": "ZeroDivisionError: division by zero"},
        {"ok": True, "result": "0"},
    ])

    class FakeResponse:
        content = "def divide(a, b):\n    return 0 if b == 0 else a / b"

    with patch.object(app, "execute_in_docker", side_effect=lambda *args: next(executions)), patch.object(
        app, "ChatOpenAI"
    ) as model:
        model.return_value.invoke.return_value = FakeResponse()
        result = app.run_agent("def divide(a, b): return a / b", "divide", {"a": 1, "b": 0})

    assert result["execution_result"]["ok"] is True
    assert result["retry_count"] == 1
    assert "0 if b == 0" in result["source_code"]


def test_failed_fixes_stop_after_three_attempts():
    from unittest.mock import Mock

    broken_model = Mock()
    broken_model.invoke.side_effect = RuntimeError("model unavailable")

    with patch.object(app, "execute_in_docker", return_value={"ok": False, "error": "still broken"}), patch.object(
        app, "ChatOpenAI", return_value=broken_model
    ):
        result = app.run_agent("def broken(): return missing_name", "broken")

    assert result["retry_count"] == app.MAX_FIX_ATTEMPTS == 3
    assert result["execution_result"]["ok"] is False
    assert broken_model.invoke.call_count == app.MAX_FIX_ATTEMPTS


def test_docker_success_removes_its_named_container():
    completed = subprocess.CompletedProcess([], 0, '{"ok": true, "result": "5"}\n', "")
    with patch.object(app, "DOCKER_IMAGE", TEST_IMAGE), patch.object(
        app.uuid, "uuid4", return_value=SimpleNamespace(hex="exec-success")
    ), patch.object(app.subprocess, "run", side_effect=_docker_responses(completed)) as run:
        result = app.execute_in_docker("def f(): return 5", "f", {})

    assert result == {"ok": True, "result": "5"}
    calls = [call.args[0] for call in run.call_args_list]
    assert calls[0][2:4] == ["--name", "agentic-bugfix-exec-success"]
    assert calls[-1][1:4] == ["rm", "--force", "container-id"]


def test_docker_timeout_kills_and_removes_only_its_container():
    with patch.object(app, "DOCKER_IMAGE", TEST_IMAGE), patch.object(
        app.uuid, "uuid4", return_value=SimpleNamespace(hex="exec-timeout")
    ), patch.object(
        app.subprocess,
        "run",
        side_effect=[
            subprocess.TimeoutExpired(["docker", "run"], 10),
            subprocess.CompletedProcess([], 0, "container-id|exec-timeout\n", ""),
            subprocess.CompletedProcess([], 0, "container-id\n", ""),
            subprocess.CompletedProcess([], 0, "container-id\n", ""),
        ],
    ) as run:
        result = app.execute_in_docker("def f(): pass", "f", {}, timeout=10)

    assert result["ok"] is False
    calls = [call.args[0] for call in run.call_args_list]
    assert calls[2] == ["docker", "kill", "container-id"]
    assert calls[3] == ["docker", "rm", "--force", "container-id"]


def test_docker_execution_failure_still_removes_its_container():
    failed = subprocess.CompletedProcess([], 125, "", "container startup failed")
    with patch.object(app, "DOCKER_IMAGE", TEST_IMAGE), patch.object(
        app.uuid, "uuid4", return_value=SimpleNamespace(hex="exec-failure")
    ), patch.object(app.subprocess, "run", side_effect=_docker_responses(failed)) as run:
        result = app.execute_in_docker("def f(): pass", "f", {})

    assert result["ok"] is False
    assert result["error"] == "container startup failed"
    assert run.call_args_list[-1].args[0][1:4] == ["rm", "--force", "container-id"]
