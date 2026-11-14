# Agentic Bug Fixing – OpenAI version

A classroom demonstration of a LangGraph repair loop. It executes a Python function, records its result or error, asks an OpenAI ChatOpenAI model to rewrite a failing function, and tries the new version. It makes at most **3 fixing attempts**.

## Project files

- `agentic_bug_fixing_openai.py`: State, `FIX_PROMPT`, Docker execution adapter, graph nodes, router, and compiled StateGraph.
- `app.py`: Gradio interface for source code, function name, JSON arguments, status, and original/current code comparison.
- `test_openai.py`: basic success, runtime-repair, and failed-fix limit tests. Tests mock both Docker execution and OpenAI calls.
- `open.py`: original minimal OpenAI API example retained from the starter project.

## Setup

Use Python 3.10 or newer. Install Docker Desktop or Docker Engine and ensure the Docker CLI can run containers. Copy `.env.example` to `.env`, then set `OPENAI_API_KEY` and, optionally, `OPENAI_MODEL`. The default model is `gpt-4o-mini`. Keep `.env` private; it is ignored by Git.

Install dependencies from `requirements.txt`, then start the interface with:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Set `BUGFIX_DOCKER_IMAGE` in `.env` to an immutable reference of the form `python:3.12-slim@sha256:<64 hex characters>`. Obtain the digest from the locally available image with `docker image inspect python:3.12-slim --format '{{json .RepoDigests}}'`; the runner rejects floating tags. The first execution may require Docker to obtain the pinned image. Arguments are a JSON object for keyword arguments or an array for positional arguments. The function name must identify a function defined in the submitted source.

Run tests with `pytest`.

## Code execution boundary and limitations

The execution node uses Python `exec()` to define and call the submitted function, but only inside a Docker container. There is deliberately no fallback that executes submitted code in the Gradio or agent process. Each execution gets a UUID-based name and ownership label. The runner verifies the container ID and label before cleanup; it removes the container after success or failure, and on timeout explicitly kills it and then force-removes it. The container is run without network access, with a read-only root filesystem, limited memory, CPU and process count, dropped Linux capabilities, and a 10-second host-side timeout. Docker must be available or execution stops with an infrastructure error.

These settings reduce risk for a local classroom demonstration; **they do not make execution safe for hostile or sensitive code**. Container isolation depends on a correctly configured and patched Docker host and kernel, and this project has no hardened sandbox, syscall policy, per-user isolation, or production abuse controls. Do not expose the Gradio app to untrusted users or run code containing sensitive data. Use a dedicated sandbox service for real untrusted-code workloads.
