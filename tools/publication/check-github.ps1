param([string]$RepoRoot)
$ErrorActionPreference='Stop'
$env:GIT_TERMINAL_PROMPT='0'
$env:GCM_INTERACTIVE='Never'
$credentialRequest = @('protocol=https','host=github.com','username=Eljo-M','') -join [char]10
$credentialLines = $credentialRequest | git -C $RepoRoot -c credential.https://github.com.username=Eljo-M credential fill
if ($LASTEXITCODE -ne 0) { throw 'Eljo-M credential unavailable' }
$credentialMap = @{}
foreach ($entry in $credentialLines) { $pair = $entry -split '=',2; if ($pair.Length -eq 2) { $credentialMap[$pair[0]]=$pair[1] } }
if (-not $credentialMap.password) { throw 'No GitHub credential returned' }
$headers=@{Authorization='Bearer '+$credentialMap.password; Accept='application/vnd.github+json'; 'X-GitHub-Api-Version'='2022-11-28'; 'User-Agent'='KernelBench-local-verification'}
$identity=Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers
if ($identity.login -cne 'Eljo-M') { throw ('Unexpected GitHub identity: '+$identity.login) }
$repo=Invoke-RestMethod -Uri 'https://api.github.com/repos/Eljo-M/kernelbench-rl' -Headers $headers
if (-not $repo.permissions.push) { throw 'Eljo-M cannot push this repository' }
Write-Output ('Authenticated GitHub login: '+$identity.login)
Write-Output ('Repository: '+$repo.full_name+'; push permission: '+$repo.permissions.push)
