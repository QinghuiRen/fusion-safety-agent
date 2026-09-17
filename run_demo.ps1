param(
  [string]$Python = "python",
  [string]$Town = "Town01",
  [double]$Duration = 35,
  [int]$Npc = 8,
  [ValidateSet("shadow", "active")][string]$Supervisor = "active"
)
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python (Join-Path $Root "fusion_safety_agent.py") demo `
  --config (Join-Path $Root "configs\product\fusion_safety_agent.json") `
  --town $Town --duration $Duration --npc $Npc --supervisor-mode $Supervisor
exit $LASTEXITCODE
