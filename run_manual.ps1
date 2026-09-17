param(
  [string]$Python = "python",
  [string]$Town = "Town01",
  [double]$Duration = 0,
  [int]$Npc = 8,
  [ValidateSet("shadow", "active")][string]$Supervisor = "shadow"
)
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python (Join-Path $Root "fusion_safety_agent.py") manual `
  --config (Join-Path $Root "configs\product\fusion_safety_agent.json") `
  --town $Town --duration $Duration --npc $Npc --manual-supervisor $Supervisor
exit $LASTEXITCODE
