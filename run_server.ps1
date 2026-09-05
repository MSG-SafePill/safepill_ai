$ErrorActionPreference = "Stop"

function Test-PythonCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command
    )

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        return $false
    }

    if ($Command -eq "py") {
        try {
            & py -3.11 --version *> $null
        } catch {
            return $false
        }
        return $LASTEXITCODE -eq 0
    }

    try {
        & $Command --version *> $null
    } catch {
        return $false
    }
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    try {
        & $Command -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
    } catch {
        return $false
    }
    return $LASTEXITCODE -eq 0
}

if (-not (Test-Path ".venv")) {
    $pythonCommand = $null
    foreach ($candidate in @("py", "python", "python3")) {
        if (Test-PythonCommand $candidate) {
            $pythonCommand = $candidate
            break
        }
    }

    if (-not $pythonCommand) {
        throw "Python 3.11 or newer is required. Install Python and make sure py, python, or python3 is available in PATH."
    }

    if ($pythonCommand -eq "py") {
        py -3.11 -m venv .venv
    } else {
        & $pythonCommand -m venv .venv
    }
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python_service\requirements.txt
python -m uvicorn python_service.app:app --reload --host 0.0.0.0 --port 8000
