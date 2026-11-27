param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][string]$ManifestPath,
    [Parameter(Mandatory=$true)][string]$Tag,
    [Parameter(Mandatory=$true)][string]$TargetCommit,
    [Parameter(Mandatory=$true)][string]$Title,
    [Parameter(Mandatory=$true)][string]$BodyPath,
    [Parameter(Mandatory=$true)][string]$ReceiptPath
)
$ErrorActionPreference='Stop'
$env:GIT_TERMINAL_PROMPT='0'
$env:GCM_INTERACTIVE='Never'
if ($TargetCommit -notmatch '^[a-f0-9]{40}$') { throw 'A full source commit is required' }
$commitIdentity=git -C $RepoRoot show -s --format='%an <%ae>|%cn <%ce>' $TargetCommit
if ($LASTEXITCODE -ne 0 -or $commitIdentity -cne 'Eljo-M <mustafajeljo@gmail.com>|Eljo-M <mustafajeljo@gmail.com>') { throw 'Unexpected source contribution identity' }
$requestText=@('protocol=https','host=github.com','username=Eljo-M','') -join [char]10
$credentialLines=$requestText | git -C $RepoRoot -c credential.https://github.com.username=Eljo-M credential fill
if ($LASTEXITCODE -ne 0) { throw 'Eljo-M authentication unavailable' }
$credentialMap=@{}
foreach($entry in $credentialLines) { $pair=$entry -split '=',2; if($pair.Length -eq 2) {$credentialMap[$pair[0]]=$pair[1]} }
if(-not $credentialMap.password) { throw 'Eljo-M credential missing' }
$headers=@{Authorization='Bearer '+$credentialMap.password;Accept='application/vnd.github+json';'X-GitHub-Api-Version'='2022-11-28';'User-Agent'='KernelBench-release-publisher'}
$api='https://api.github.com/repos/Eljo-M/kernelbench-rl'
$identity=Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers
if($identity.login -cne 'Eljo-M') {throw 'Authenticated GitHub login must be Eljo-M'}
$repo=Invoke-RestMethod -Uri $api -Headers $headers
if(-not $repo.permissions.push -or $repo.private) {throw 'Expected authorized public project repository'}
Write-Output 'Verified authenticated GitHub login and source author/committer: Eljo-M'
$manifest=Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$assets=@($manifest.assets)
if($assets.Count -eq 0) {throw 'No concrete release assets'}
$assetNames=@($assets | ForEach-Object {[System.IO.Path]::GetFileName($_.path)})
if(@($assetNames | Group-Object -CaseSensitive | Where-Object {$_.Count -gt 1}).Count){throw 'Duplicate local asset basenames'}
foreach($asset in $assets) {
    $file=Get-Item -LiteralPath $asset.path
    if($file.Length -ne $asset.bytes -or $file.Length -ge 2GB) {throw ('Invalid asset size: '+$file.Name)}
    if((Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower() -cne $asset.sha256) {throw ('Asset hash changed: '+$file.Name)}
}
function Get-ReleaseTagCommit {
    $reference=$null
    try {$reference=Invoke-RestMethod -Uri ($api+'/git/ref/tags/'+[uri]::EscapeDataString($Tag)) -Headers $headers}
    catch {if([int]$_.Exception.Response.StatusCode -eq 404){return $null}; throw}
    $object=$reference.object
    for($level=0;$level -lt 10;$level++) {
        if($object.type -ceq 'commit'){return $object.sha}
        if($object.type -cne 'tag'){throw 'Release tag does not resolve to a commit'}
        $annotated=Invoke-RestMethod -Uri ($api+'/git/tags/'+$object.sha) -Headers $headers
        $object=$annotated.object
    }
    throw 'Release tag nesting exceeds supported depth'
}
$tagCommit=Get-ReleaseTagCommit
if($tagCommit -and $tagCommit -cne $TargetCommit){throw 'Existing tag points to a different source commit'}
$release=$null
try {$release=Invoke-RestMethod -Uri ($api+'/releases/tags/'+[uri]::EscapeDataString($Tag)) -Headers $headers}
catch {if([int]$_.Exception.Response.StatusCode -ne 404){throw}}
if($release) {
    if(-not $tagCommit -and (-not $release.draft -or $release.target_commitish -cne $TargetCommit)){throw 'Existing release lacks exact source binding'}
} else {
    $payload=@{tag_name=$Tag;target_commitish=$TargetCommit;name=$Title;body=(Get-Content -Raw -LiteralPath $BodyPath);draft=$true;prerelease=$false} | ConvertTo-Json -Depth 4
    $release=Invoke-RestMethod -Method Post -Uri ($api+'/releases') -Headers $headers -ContentType 'application/json' -Body $payload
}
$existing=@(Invoke-RestMethod -Uri ($api+'/releases/'+$release.id+'/assets?per_page=100') -Headers $headers)
$client=[System.Net.Http.HttpClient]::new()
$client.Timeout=[TimeSpan]::FromHours(2)
$client.DefaultRequestHeaders.Authorization=[System.Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer',$credentialMap.password)
$client.DefaultRequestHeaders.UserAgent.ParseAdd('KernelBench-release-publisher')
$client.DefaultRequestHeaders.Accept.ParseAdd('application/vnd.github+json')
$client.DefaultRequestHeaders.Add('X-GitHub-Api-Version','2022-11-28')
$receipt=[ordered]@{tag=$Tag;source_commit=$TargetCommit;release_url=$release.html_url;authenticated_login='Eljo-M';published=(-not $release.draft);assets=@()}
try {
    foreach($asset in $assets) {
        $name=[System.IO.Path]::GetFileName($asset.path)
        $previous=@($existing | Where-Object {$_.name -ceq $name})
        if($previous.Count -gt 1){throw ('Duplicate remote asset: '+$name)}
        if($previous.Count -eq 1){
            $uploaded=$previous[0]
            if($uploaded.size -ne $asset.bytes -or $uploaded.digest -cne ('sha256:'+$asset.sha256)){throw ('Existing remote asset digest mismatch: '+$name)}
            Write-Output ('Verified existing asset: '+$name)
        } else {
            Write-Output ('Uploading '+$name+' ('+$asset.bytes+' bytes)')
            $stream=[System.IO.File]::OpenRead($asset.path)
            $content=[System.Net.Http.StreamContent]::new($stream,1MB)
            $content.Headers.ContentType=[System.Net.Http.Headers.MediaTypeHeaderValue]::new('application/octet-stream')
            $content.Headers.ContentLength=$stream.Length
            try {
                $uri='https://uploads.github.com/repos/Eljo-M/kernelbench-rl/releases/'+$release.id+'/assets?name='+[uri]::EscapeDataString($name)
                $response=$client.PostAsync($uri,$content).GetAwaiter().GetResult()
                $response.EnsureSuccessStatusCode() | Out-Null
                $uploaded=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
            } finally {$content.Dispose();$stream.Dispose()}
            if($uploaded.size -ne $asset.bytes -or $uploaded.digest -cne ('sha256:'+$asset.sha256)){throw ('Uploaded asset digest mismatch: '+$name)}
            Write-Output ('Verified remote SHA-256: '+$name)
        }
        $receipt.assets+=@{name=$name;bytes=$uploaded.size;sha256=$asset.sha256;url=$uploaded.browser_download_url;asset_id=$uploaded.id}
        $receipt | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
    }
    if($release.draft){
        $publish=@{draft=$false;make_latest='true'} | ConvertTo-Json
        $release=Invoke-RestMethod -Method Patch -Uri ($api+'/releases/'+$release.id) -Headers $headers -ContentType 'application/json' -Body $publish
    }
    if($release.draft -or (Get-ReleaseTagCommit) -cne $TargetCommit){throw 'Published release source binding failed'}
    $receipt.published=$true
    $receipt.release_url=$release.html_url
    $receipt | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
} finally {$client.Dispose();$credentialMap.Clear()}
Write-Output ('RELEASE_COMPLETE '+$release.html_url)
