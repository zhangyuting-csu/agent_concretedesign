$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:NATUREML_HOME = Join-Path $root 'testML'
if (-not $env:PORT) { $env:PORT = '8060' }
Set-Location -LiteralPath $root
python .\server.py
