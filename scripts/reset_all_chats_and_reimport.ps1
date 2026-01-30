param(
  [string]$BackupDir = "",
  [string]$DefaultPlatform = "taobao",
  [switch]$DryRun,
  [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Checking docker compose..." -ForegroundColor Cyan
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "docker is not installed or not in PATH"
}

$composeArgs = @("compose")

if (-not $NoBuild) {
  Write-Host "[2/3] Ensuring containers are up (may rebuild)..." -ForegroundColor Cyan
  docker @composeArgs up -d --build | Out-Null
} else {
  Write-Host "[2/3] Ensuring containers are up..." -ForegroundColor Cyan
  docker @composeArgs up -d | Out-Null
}

Write-Host "[3/3] Resetting chat tables and re-importing from backups..." -ForegroundColor Cyan

$cmd = @("python", "-m", "tools.reset_chat_and_reimport_from_backup")
if ($BackupDir -and $BackupDir.Trim().Length -gt 0) {
  $cmd += @("--backup-dir", $BackupDir)
}
if ($DefaultPlatform -and $DefaultPlatform.Trim().Length -gt 0) {
  $cmd += @("--default-platform", $DefaultPlatform)
}
if ($DryRun) {
  $cmd += "--dry-run"
}

docker @composeArgs exec -T app @cmd

Write-Host "\nDone." -ForegroundColor Green
Write-Host "Tip: if you want a preview, run: .\scripts\reset_all_chats_and_reimport.ps1 -DryRun" -ForegroundColor DarkGray
