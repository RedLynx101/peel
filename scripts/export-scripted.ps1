$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$out = "$repo\artifacts\data\replays\scripted-door-0.json"
& "$repo\.venv\Scripts\python.exe" -m peel.replay --stage door --seed 0 --output $out
