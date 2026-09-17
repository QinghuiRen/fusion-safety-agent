param([string]$Python = "python")
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python (Join-Path $Root "fusion_safety_agent.py") doctor `
  --config (Join-Path $Root "configs\product\fusion_safety_agent.json")
exit $LASTEXITCODE
