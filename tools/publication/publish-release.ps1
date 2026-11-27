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
if(Test-Path -LiteralPath $ReceiptPath){throw 'Preserve the earlier release receipt; select a new ReceiptPath'}
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
if($manifest.builder -cne 'kernelbench_retry_export_v1' -or $manifest.source_commit -cne $TargetCommit){throw 'Upload manifest does not bind this exact publication source'}
$assets=@($manifest.assets)
if($assets.Count -eq 0) {throw 'No concrete release assets'}
$assetNames=@($assets | ForEach-Object {[System.IO.Path]::GetFileName($_.path)})
if(@($assetNames | Group-Object -CaseSensitive | Where-Object {$_.Count -gt 1}).Count){throw 'Duplicate local asset basenames'}
foreach($asset in $assets) {
    if($asset.sha256 -cnotmatch '^[a-f0-9]{64}$' -or $asset.bytes -le 0){throw 'Invalid asset digest/byte metadata'}
    $file=Get-Item -LiteralPath $asset.path
    if($file.Length -ne $asset.bytes -or $file.Length -ge 2GB) {throw ('Invalid asset size: '+$file.Name)}
    if((Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower() -cne $asset.sha256) {throw ('Asset hash changed: '+$file.Name)}
}
$main=Invoke-RestMethod -Uri ($api+'/commits/main') -Headers $headers
if($main.sha -cne $TargetCommit){throw 'Public main differs from reviewed publication source'}
$deliveryAsset=@($assets | Where-Object {[IO.Path]::GetFileName($_.path) -ceq 'RETRY_DELIVERY_MANIFEST.json'})
if($deliveryAsset.Count -ne 1){throw 'Expected one validated retry delivery manifest asset'}
$delivery=Get-Content -Raw -LiteralPath $deliveryAsset[0].path | ConvertFrom-Json
if($delivery.release_source_commit -cne $TargetCommit -or $delivery.publication_byte_verification.verified -ne $true -or $delivery.publication_byte_verification.clone_check_performed -ne $true){throw 'Delivery manifest lacks exact-source completed clone verification'}
$ciMatch=[regex]::Match([string]$delivery.ci.url,'^https://github.com/Eljo-M/kernelbench-rl/actions/runs/([0-9]+)$')
if(-not $ciMatch.Success){throw 'Delivery manifest final CI URL missing'}
$ciRunId=$ciMatch.Groups[1].Value
$ci=Invoke-RestMethod -Uri ($api+'/actions/runs/'+$ciRunId) -Headers $headers
if($ci.head_sha -cne $TargetCommit -or $ci.head_branch -cne 'main' -or $ci.name -cne 'KernelBench-RL CI' -or $ci.status -cne 'completed' -or $ci.conclusion -cne 'success'){throw 'Final publication CI is not successful for this exact main source'}
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
if(-not $release.draft){
    foreach($asset in $assets){
        $name=[IO.Path]::GetFileName($asset.path)
        if(@($existing | Where-Object {$_.name -ceq $name}).Count -ne 1){throw 'A published release has an incomplete asset set; preserve it and use a new versioned draft'}
    }
}
$client=[System.Net.Http.HttpClient]::new()
$client.Timeout=[TimeSpan]::FromHours(2)
$client.DefaultRequestHeaders.Authorization=[System.Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer',$credentialMap.password)
$client.DefaultRequestHeaders.UserAgent.ParseAdd('KernelBench-release-publisher')
$client.DefaultRequestHeaders.Accept.ParseAdd('application/vnd.github+json')
$client.DefaultRequestHeaders.Add('X-GitHub-Api-Version','2022-11-28')
$receipt=[ordered]@{schema_version='kernelbench_release_verification_v1';tag=$Tag;source_commit=$TargetCommit;release_url=$release.html_url;authenticated_login='Eljo-M';published=$false;verified=$false;release_was_already_public=(-not $release.draft);ci_url=$ci.html_url;expected_assets=$assets.Count;assets=@();started_utc=[DateTime]::UtcNow.ToString('o')}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
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
        if($uploaded.state -cne 'uploaded'){throw ('Remote asset is not fully uploaded: '+$name)}
        $receipt.assets+=@{name=$name;bytes=$uploaded.size;sha256=$asset.sha256;server_digest=$uploaded.digest;url=$uploaded.browser_download_url;asset_id=$uploaded.id}
        $receipt | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
    }
    foreach($asset in $receipt.assets){
        $remote=Invoke-RestMethod -Uri ($api+'/releases/assets/'+$asset.asset_id) -Headers $headers
        if($remote.state -cne 'uploaded' -or $remote.size -ne $asset.bytes -or $remote.digest -cne ('sha256:'+$asset.sha256)){throw ('Final server asset verification failed: '+$asset.name)}
    }
    if($release.draft){
        $publish=@{draft=$false;make_latest='true'} | ConvertTo-Json
        $release=Invoke-RestMethod -Method Patch -Uri ($api+'/releases/'+$release.id) -Headers $headers -ContentType 'application/json' -Body $publish
    }
    if($release.draft -or (Get-ReleaseTagCommit) -cne $TargetCommit){throw 'Published release source binding failed'}
    $receipt.published=$true
    $receipt.verified=$true
    $receipt.completed_utc=[DateTime]::UtcNow.ToString('o')
    $receipt.release_url=$release.html_url
    $receipt | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
} catch {
    $receipt.published=$false;$receipt.verified=$false;$receipt.error=$_.Exception.Message
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
    throw
} finally {$client.Dispose();$credentialMap.Clear()}
Write-Output ('RELEASE_COMPLETE '+$release.html_url)
