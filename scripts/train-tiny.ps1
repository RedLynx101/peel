$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& "$repo\.venv\Scripts\python.exe" -m peel.train --config "$repo\configs\tiny.json" --run-dir "$repo\runs\tiny"
