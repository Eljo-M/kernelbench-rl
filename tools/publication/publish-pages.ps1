param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][string]$SourceCommit,
    [Parameter(Mandatory=$true)][string]$ReceiptPath
)
$ErrorActionPreference='Stop'
if($SourceCommit -notmatch '^[a-f0-9]{40}$'){throw 'A full source commit is required'}
$author=git -C $RepoRoot show -s --format='%an <%ae>|%cn <%ce>' $SourceCommit
if($LASTEXITCODE -ne 0 -or $author -cne 'Eljo-M <mustafajeljo@gmail.com>|Eljo-M <mustafajeljo@gmail.com>'){throw 'Unexpected source author/committer'}
$env:GIT_TERMINAL_PROMPT='0'; $env:GCM_INTERACTIVE='Never'
$requestText=@('protocol=https','host=github.com','username=Eljo-M','') -join [char]10
$credentialLines=$requestText | git -C $RepoRoot -c credential.https://github.com.username=Eljo-M credential fill
if($LASTEXITCODE -ne 0){throw 'Eljo-M GitHub authentication unavailable'}
$credentialMap=@{}
foreach($entry in $credentialLines){$pair=$entry -split '=',2;if($pair.Length -eq 2){$credentialMap[$pair[0]]=$pair[1]}}
if(-not $credentialMap.password){throw 'GitHub credential missing'}
$headers=@{Authorization='Bearer '+$credentialMap.password;Accept='application/vnd.github+json';'X-GitHub-Api-Version'='2022-11-28';'User-Agent'='KernelBench-publication'}
$api='https://api.github.com/repos/Eljo-M/kernelbench-rl'
try {
    $user=Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers
    if($user.login -cne 'Eljo-M'){throw 'Authenticated account must be Eljo-M'}
    $repo=Invoke-RestMethod -Uri $api -Headers $headers
    if($repo.private -or -not $repo.permissions.admin){throw 'Expected Eljo-M administrative access to the existing public repository'}
    $main=Invoke-RestMethod -Uri ($api+'/commits/main') -Headers $headers
    if($main.sha -cne $SourceCommit){throw 'Public main does not match the reviewed source commit'}
    foreach($path in @('docs/index.html','docs/paper.html','docs/assets/kernelbench-rl-paper.pdf','docs/.nojekyll')){
        git -C $RepoRoot cat-file -e ($SourceCommit+':'+$path)
        if($LASTEXITCODE -ne 0){throw ('Publication input missing in source commit: '+$path)}
    }
    $pages=$null
    try{$pages=Invoke-RestMethod -Uri ($api+'/pages') -Headers $headers}
    catch{if([int]$_.Exception.Response.StatusCode -ne 404){throw}}
    if($pages){
        if($pages.cname -or $pages.source.branch -cne 'main' -or $pages.source.path -cne '/docs'){
            throw 'Existing Pages configuration differs; preserve it for review instead of replacing it'
        }
        Write-Output 'Reusing the matching main/docs Pages configuration'
    } else {
        $payload=@{source=@{branch='main';path='/docs'};build_type='legacy'} | ConvertTo-Json -Depth 4
        $pages=Invoke-RestMethod -Method Post -Uri ($api+'/pages') -Headers $headers -ContentType 'application/json' -Body $payload
        Write-Output 'Created public GitHub Pages publication from the reviewed main/docs source'
    }
    $receipt=[ordered]@{authenticated_login='Eljo-M';source_commit=$SourceCommit;url=$pages.html_url;status=$pages.status;source=$pages.source;created_or_verified_utc=[DateTime]::UtcNow.ToString('o');note='Publication status and public content must be verified after the Pages build.'}
    $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
    Write-Output ('PAGES_PUBLICATION '+$pages.html_url+' status='+$pages.status)
} finally {$credentialMap.Clear()}
