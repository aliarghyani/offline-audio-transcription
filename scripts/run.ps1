param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $CliArgs
)

Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$uv = Get-Command uv -ErrorAction Stop
& $uv.Source run --no-sync --project $repoRoot session-pipeline @CliArgs
exit $LASTEXITCODE
