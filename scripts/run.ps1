[CmdletBinding(PositionalBinding = $false)]
param(
  [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
  [string[]] $CliArgs = @(),
  [ValidateSet('fast', 'quality', 'full')]
  [string] $Profile,
  [ValidateSet('auto-prefer-retry', 'conservative', 'human')]
  [string] $ReviewPolicy
)

Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$uv = Get-Command uv -ErrorAction Stop
if ($CliArgs.Count -eq 0) {
  Write-Error 'Supply a command, for example: transcribe .\sessions\session5.m4a -Profile full'
  exit 2
}
if ($Profile) { $CliArgs += @('--profile', $Profile) }
if ($ReviewPolicy) { $CliArgs += @('--review-policy', $ReviewPolicy) }
# Offline uv prevents package resolution/downloads during normal use. No elevation.
& $uv.Source run --offline --no-sync --project $repoRoot session-pipeline @CliArgs
exit $LASTEXITCODE
