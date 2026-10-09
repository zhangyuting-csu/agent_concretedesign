# Demo code package

This is a **framework-only** release. It contains the web UI, API code, machine-learning platform code, and startup script. It deliberately contains **no dataset, literature abstract or full text, standard text or extracted constraints, knowledge-graph routes or evidence, benchmark history, trained models, credentials, or user records**.

## Requirements

- Windows 10/11, PowerShell, Python 3.11-3.13 recommended.
- Internet access is needed to install packages and load the current frontend CDN dependencies (MathJax, Vue, Axios, Tailwind CSS, Font Awesome). For an offline demo, these frontend libraries must first be hosted locally.
- Optional: an OpenAI-compatible local LLM service, default `http://localhost:1234/v1`, for agent inference.

## Run

1. Extract the ZIP into a new, writable folder.
2. In PowerShell, run `python -m venv .venv`, then `.\.venv\Scripts\Activate.ps1`.
3. Run `python -m pip install -r requirements.txt`.
4. Run `.\运行演示.ps1` (or `python server.py`). The default port is 8060; set `$env:PORT='8061'` before startup if needed.
5. Open `http://127.0.0.1:8060/`.

## What works without private data

The application shell, navigation, API structure, Skill defaults, and machine-learning upload workflow are present. Evidence panels, KG retrieval, standard constraints, and trained-model predictions have **no source data in this package**. They may show empty states or report missing inputs; this is intentional and is not a complete scientific-results demo.

To run the full research workflow, the owner must separately provide authorized materials and configure `STANDARD_PDF_DIR`, `NATUREML_HOME` (if the bundled `testML` directory is moved), `LM_STUDIO_URL`, and model files as appropriate. Do not put sensitive datasets or credentials into a public submission. Python 3.14 may not support all optional machine-learning packages.

## Package boundary

Only source-code and frontend files listed by the packaging script are included. The `data/`, `evidence/`, `skills/` source folders and all prior visualization outputs are excluded. The server can create empty runtime directories after launch; such generated files are not part of this ZIP.
