"""Gradio interface for the classroom Agentic Bug Fixing workflow."""

import json

import gradio as gr

from agentic_bug_fixing_openai import MAX_FIX_ATTEMPTS, run_agent


def run_from_ui(source_code: str, function_name: str, arguments_json: str):
    try:
        arguments = json.loads(arguments_json or "{}")
        if not isinstance(arguments, (dict, list)):
            raise ValueError("Arguments must be a JSON object (keyword arguments) or array (positional arguments).")
        state = run_agent(source_code, function_name.strip(), arguments)
        result = state.get("execution_result", {})
        if result.get("ok"):
            status = f"Success after {state.get('retry_count', 0)} fix attempt(s). Result: {result.get('result')}"
        else:
            status = f"Stopped after {state.get('retry_count', 0)} fix attempt(s).\n{state.get('error_details', result.get('error', 'Unknown error'))}"
        return status, state.get("original_source_code", source_code), state.get("source_code", source_code)
    except Exception as exc:
        return f"Input or workflow error: {exc}", source_code, source_code


with gr.Blocks(title="Agentic Bug Fixing – OpenAI version") as demo:
    gr.Markdown(
        "# Agentic Bug Fixing – OpenAI version\n"
        "Enter a Python function and its arguments. LangGraph runs it, asks OpenAI "
        f"for fixes after runtime errors, and stops after at most {MAX_FIX_ATTEMPTS} fixes.\n\n"
        "**Local classroom demo:** Docker must be installed and running. Submitted code still "
        "uses Python `exec()` inside a constrained container; this is not a hardened sandbox. "
        "Do not submit sensitive or hostile code."
    )
    with gr.Row():
        source = gr.Code(label="Python function", language="python", value="def add(a, b):\n    return a + b\n")
        function = gr.Textbox(label="Function name", value="add")
    arguments = gr.Textbox(label="Arguments as JSON", value='{"a": 2, "b": 3}')
    run_button = gr.Button("Run agent", variant="primary")
    status = gr.Textbox(label="Execution status", lines=4)
    with gr.Row():
        original = gr.Code(label="Original code", language="python")
        corrected = gr.Code(label="Current / corrected code", language="python")
    run_button.click(run_from_ui, [source, function, arguments], [status, original, corrected])


if __name__ == "__main__":
    demo.launch()
