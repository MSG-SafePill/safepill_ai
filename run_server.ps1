$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

.\.venv\Scripts\Activate.ps1
pip install -r python_service\requirements.txt
uvicorn python_service.app:app --reload --port 8000
