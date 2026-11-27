param([Parameter(Mandatory=$true)][string]$Commit,[string]$OutputPath='',
      [string]$RepoRoot='',[string]$Branch='',[switch]$RequireSuccess)
$ErrorActionPreference='Stop'
if ($Commit -notmatch '^[0-9a-f]{40}$') { throw 'Expected exact source SHA' }
if(-not $RepoRoot){
    $candidate=[IO.DirectoryInfo]::new($PSScriptRoot)
    while($candidate -and -not (Test-Path -LiteralPath (Join-Path $candidate.FullName 'rl_trainer/experiment_config.py'))){$candidate=$candidate.Parent}
    $RepoRoot=if($candidate){$candidate.FullName}else{[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../outputs/kernelbench-rl'))}
}
if($OutputPath -and (Test-Path -LiteralPath $OutputPath)){throw 'Preserve earlier CI evidence; select a new OutputPath'}
. "$PSScriptRoot/check-github.ps1" -RepoRoot $RepoRoot
$runs=Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/Eljo-M/kernelbench-rl/actions/runs?head_sha=$Commit&per_page=30"
$summary=@($runs.workflow_runs | Where-Object { $_.head_sha -ceq $Commit -and (-not $Branch -or $_.head_branch -ceq $Branch) } | ForEach-Object {
    [ordered]@{ id=$_.id; name=$_.name; head_sha=$_.head_sha; head_branch=$_.head_branch; status=$_.status; conclusion=$_.conclusion; html_url=$_.html_url; created_at=$_.created_at; updated_at=$_.updated_at }
})
$successful=@($summary | Where-Object {$_.name -ceq 'KernelBench-RL CI' -and $_.status -ceq 'completed' -and $_.conclusion -ceq 'success'})
$payload=[ordered]@{schema_version='kernelbench_github_ci_receipt_v1';authenticated_login=$identity.login;requested_source_commit=$Commit;requested_branch=$Branch;verified=($successful.Count -gt 0);observed_utc=[DateTime]::UtcNow.ToString('o');runs=$summary}
$json=$payload | ConvertTo-Json -Depth 8
if ($OutputPath) { [IO.File]::WriteAllText([IO.Path]::GetFullPath($OutputPath),$json+[Environment]::NewLine,[Text.UTF8Encoding]::new($false)) }
Write-Output $json
$credentialMap.Clear()
if($RequireSuccess -and -not $payload.verified){throw 'Exact-source CI has not completed successfully; pending/failure receipt retained'}
