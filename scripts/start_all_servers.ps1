# Starts all six ACP servers, each in its own PowerShell window.
# Run from the project root:  .\scripts\start_all_servers.ps1
#
# After they're up, run the client in this terminal:  uv run acp_client.py

$ErrorActionPreference = "Stop"

# Resolve the project root (parent of this scripts/ folder) so the script works
# regardless of where it's invoked from.
$projectRoot = Split-Path -Parent $PSScriptRoot

# Each server: a friendly title and its file.
$servers = @(
    @{ Title = "profile_analyzer :8000";     File = "profile_acp_server.py" },
    @{ Title = "job_searcher :8001";         File = "job_search_acp_server.py" },
    @{ Title = "job_ranker :8002";           File = "ranking_acp_server.py" },
    @{ Title = "resume_tailor :8003";        File = "resume_acp_server.py" },
    @{ Title = "cover_letter_writer :8004";  File = "cover_letter_acp_server.py" },
    @{ Title = "interview_coach :8005";      File = "interview_acp_server.py" }
)

foreach ($s in $servers) {
    Write-Host "Starting $($s.Title) -> $($s.File)"
    # Open a new PowerShell window, cd to the project root, and run the server with uv.
    # -NoExit keeps the window open so you can read that server's logs/errors.
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location -LiteralPath '$projectRoot'; `$host.UI.RawUI.WindowTitle = '$($s.Title)'; uv run $($s.File)"
    )
    Start-Sleep -Milliseconds 800
}

Write-Host ""
Write-Host "All six servers launched in separate windows."
Write-Host "Give them a moment to boot, verify with:  .\scripts\check_agents.ps1"
Write-Host "Then run the client:  uv run acp_client.py"
