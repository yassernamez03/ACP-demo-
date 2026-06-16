# Checks that each ACP server is up and reports the agent(s) it exposes.
# Run from the project root:  .\scripts\check_agents.ps1

$ports = 8000, 8001, 8002, 8003, 8004, 8005

foreach ($port in $ports) {
    $url = "http://localhost:$port/agents"
    try {
        $resp = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 5
        $names = ($resp.agents | ForEach-Object { $_.name }) -join ", "
        if ([string]::IsNullOrWhiteSpace($names)) { $names = "(no agents registered)" }
        Write-Host "[OK]  :$port  ->  $names" -ForegroundColor Green
    }
    catch {
        Write-Host "[DOWN] :$port  ->  not reachable (start its server with 'uv run ...')" -ForegroundColor Red
    }
}
