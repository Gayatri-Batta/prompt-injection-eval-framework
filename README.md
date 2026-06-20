# Prompt Injection Safety Evaluation Platform

This project evaluates whether workplace AI agents can complete legitimate tasks while resisting malicious instructions hidden inside untrusted Slack messages, files, or tool outputs.

It combines:

- AgentDojo as a validated public prompt-injection benchmark.
- A reproducible generated workplace dataset (or your own uploaded dataset).
- Model and defense comparisons across OpenAI, Groq, and Anthropic.
- pandas metrics, DuckDB analytics, Plotly charts, and a Dash dashboard.

## Quickstart

```powershell
pip install -r requirements.txt
$env:PYTHONPATH="src"
python scripts/generate_custom_dataset.py
python dashboard/app.py
```

Open the local Dash URL printed by the final command, usually:

```text
http://127.0.0.1:8050
```

Use the **Run Experiment** tab to configure a run. Settings are grouped into three collapsible sections (only one open at a time, each summarized when collapsed):

1. **Dataset** -- pick the default generated dataset or upload your own (`.jsonl`, `.json`, or `.csv`).
2. **Models and Defenses** -- pick one or more models and one or more defenses, and (if LLM-judge scoring is on) which judge model to use.
3. **Scenario Filters** -- optionally narrow scenarios by type, attack technique, or severity (leave any empty to include everything on that axis), then set how many scenarios to sample from whatever matches.

The **Run Summary** panel above the Run Experiment button estimates the number of evaluations, API calls, and cost for your current selections before you commit to a run. Click **Run Experiment** to launch real model calls -- the **Experiment Status** panel shows live **Progress** (step-by-step checklist) and **Logs**. Once a run finishes, the app parses results, rebuilds DuckDB analytics, refreshes the report, and the dashboard's Overview, Scenario Analysis, and Run Drilldown tabs reflect the new data, labeled with the completion timestamp.

## Models and API Keys

Add the key(s) for whichever provider(s) you want to use to `.env` (see `.env.example`):

```text
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Supported providers and their models are listed in `configs/models.yaml`. You don't need every key set -- the dashboard only complains about a missing key once you select a model from that provider. Adding a new model is as simple as adding an entry to `configs/models.yaml`.

## Defenses

Four defense strategies are available, selectable per run:

- **baseline** -- no defense; the model sees the untrusted content as-is.
- **instruction_boundary** -- wraps untrusted content with an explicit system-prompt boundary instructing the model to treat it as data, not instructions.
- **spotlighting** -- marks untrusted content with inline delimiters so the model can visually distinguish it from the real task.
- **sandwich** -- repeats the original user task after the untrusted content, to counteract instructions buried at the end.

## Scoring

Responses can be scored two ways, controlled by `judging.enabled` in `configs/benchmark.yaml`:

- **LLM-as-judge** (default) -- a judge model (e.g. `gpt-4o-mini`) evaluates whether the task was completed and whether the attack succeeded. Each scenario makes one call for the response and a second call for the judge's evaluation.
- **Keyword matching** -- a fallback/cheaper mode that checks for expected/forbidden keywords in the model's output. Always computed alongside judge scores as `keyword_task_success` / `keyword_attack_success` for comparison.

## Overview Tab and Run History

Every model/defense combination launched from one click of **Run Experiment** (or one CLI invocation) is grouped together as a single "run," identified by when it was submitted. The Overview tab charts the most recent run by default, and lists every run in an expandable history -- open any run to see its own model/defense breakdown without other runs mixed in.

## Custom Datasets

Upload your own dataset as `.jsonl` (one JSON object per line), `.json` (a top-level array of objects), or `.csv`. Required and recommended fields, plus an example row, are shown in the "Dataset format for uploads" section of the Run Experiment tab. An upload that's missing required fields, has duplicate `scenario_id`s, or fails to parse is rejected with a specific error before it's used.

## AgentDojo

AgentDojo is included as an optional public benchmark path:

```powershell
pip install -r requirements-agentdojo.txt
$env:PYTHONPATH="src"
python scripts/run_benchmark.py --dataset agentdojo --model gpt-4o-mini --dry-run
```

Remove `--dry-run` to execute the AgentDojo CLI after configuring credentials.

## Metrics

- `benign_success`: task success without injected malicious content.
- `attacked_success`: task success with injected content present.
- `utility_drop`: benign success minus attacked success (shown on the dashboard as **Utility Change**).
- `attack_success_rate`: how often the malicious instruction succeeded.
- `robust_success_rate`: task succeeds and attack fails (shown on the dashboard as **Task Success Under Attack**).

The most important metric is `robust_success_rate`.
