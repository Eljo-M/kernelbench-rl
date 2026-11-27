param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][string]$SourceCommit,
    [Parameter(Mandatory=$true)][string]$ReceiptPath,
    [ValidateRange(1,45)][int]$MaxWaitSeconds=40,
    [ValidateRange(1,20)][int]$PollSeconds=10
)
$ErrorActionPreference='Stop'
if($SourceCommit -cnotmatch '^[a-f0-9]{40}$'){throw 'Exact publication source SHA required'}
$RepoRoot=[IO.Path]::GetFullPath($RepoRoot)
$author=git -C $RepoRoot show -s --format='%an <%ae>|%cn <%ce>' $SourceCommit
if($LASTEXITCODE -ne 0 -or $author -cne 'Eljo-M <mustafajeljo@gmail.com>|Eljo-M <mustafajeljo@gmail.com>'){throw 'Unexpected source author/committer'}
git -C $RepoRoot diff --quiet $SourceCommit -- docs
if($LASTEXITCODE -ne 0){throw 'Working docs differ from publication source'}
$text=git -C $RepoRoot show ($SourceCommit+':docs/project-config.json')
if($LASTEXITCODE -ne 0){throw 'Committed website configuration missing'}
$config=($text -join [char]10) | ConvertFrom-Json
$base=([string]$config.public_base_url).TrimEnd('/')
$pdf=[string]$config.paper_pdf_path
if($config.draft -ne $false -or $base -cne 'https://eljo-m.github.io/kernelbench-rl' -or -not $pdf -or $pdf.StartsWith('/') -or $pdf.Contains('..') -or $pdf.Contains(':') -or $pdf.Contains('\')){throw 'Website origin/PDF configuration is not final'}
$names=@(git -C $RepoRoot ls-tree -r --name-only $SourceCommit -- docs)
if($LASTEXITCODE -ne 0 -or -not $names.Count){throw 'No committed public docs assets'}
foreach($required in @('docs/index.html','docs/paper.html','docs/robots.txt','docs/sitemap.xml',('docs/'+$pdf))){
    if($names -cnotcontains $required){throw ('Required public asset missing: '+$required)}
}
$paperText=[IO.File]::ReadAllText((Join-Path $RepoRoot 'docs/paper.html'))
if(-not $paperText.Contains('name="citation_author" content="Eljo Mustafaj"') -or -not $paperText.Contains(('name="citation_pdf_url" content="'+$base+'/'+$pdf+'"'))){throw 'Source paper citation metadata differs from canonical author/PDF'}
foreach($html in @($paperText,[IO.File]::ReadAllText((Join-Path $RepoRoot 'docs/index.html')))){
    if($html -match '(?is)<meta\b[^>]*\bcontent="[^"]*noindex[^"]*"'){throw 'Source HTML still contains a noindex directive'}
}
$manuscriptText=git -C $RepoRoot show ($SourceCommit+':paper/manuscript.json')
if($LASTEXITCODE -ne 0){throw 'Committed manuscript metadata missing'}
$manuscript=($manuscriptText -join [char]10) | ConvertFrom-Json
foreach($field in @(@('citation_title',[string]$manuscript.title),@('citation_publication_date',([string]$manuscript.publication_date).Replace('-','/')))){
    $pattern='name="'+$field[0]+'" content="([^"]+)"'
    $match=[regex]::Match($paperText,$pattern)
    if(-not $match.Success -or [Net.WebUtility]::HtmlDecode($match.Groups[1].Value) -cne $field[1]){throw ('Source paper metadata differs: '+$field[0])}
}
$robots=[IO.File]::ReadAllText((Join-Path $RepoRoot 'docs/robots.txt'))
if($robots -match '(?im)^\s*Disallow:\s*/\s*$'){throw 'Source robots.txt blocks indexing'}
[xml]$sitemap=[IO.File]::ReadAllText((Join-Path $RepoRoot 'docs/sitemap.xml'))
$locations=@($sitemap.SelectNodes('//*[local-name()="loc"]') | ForEach-Object {$_.InnerText})
if($locations -cnotcontains ($base+'/') -or $locations -cnotcontains ($base+'/paper.html')){throw 'Source sitemap lacks canonical project/paper URLs'}
$receipt=[ordered]@{schema_version='kernelbench_pages_public_verification_v1';verified=$false;publication_complete=$false;source_commit=$SourceCommit;url=$base;attempts=@()}
if(Test-Path -LiteralPath $ReceiptPath){
    $old=Get-Content -Raw -LiteralPath $ReceiptPath | ConvertFrom-Json
    if($old.schema_version -cne $receipt.schema_version -or $old.source_commit -cne $SourceCommit){throw 'Existing receipt belongs to a different verification/source'}
    $receipt.attempts=@($old.attempts)
}
$credentialMap=@{};$client=$null
try{
    $env:GIT_TERMINAL_PROMPT='0';$env:GCM_INTERACTIVE='Never'
    $requestText=@('protocol=https','host=github.com','username=Eljo-M','') -join [char]10
    $lines=$requestText | git -C $RepoRoot -c credential.https://github.com.username=Eljo-M credential fill
    if($LASTEXITCODE -ne 0){throw 'Eljo-M authentication unavailable'}
    foreach($line in $lines){$pair=$line -split '=',2;if($pair.Length -eq 2){$credentialMap[$pair[0]]=$pair[1]}}
    if(-not $credentialMap.password){throw 'GitHub credential missing'}
    $headers=@{Authorization='Bearer '+$credentialMap.password;Accept='application/vnd.github+json';'X-GitHub-Api-Version'='2022-11-28';'User-Agent'='KernelBench-Pages-verifier'}
    $user=Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers -TimeoutSec 15
    if($user.login -cne 'Eljo-M'){throw 'Authenticated account must be Eljo-M'}
    $receipt.authenticated_login='Eljo-M'
    $api='https://api.github.com/repos/Eljo-M/kernelbench-rl'
    $main=Invoke-RestMethod -Uri ($api+'/commits/main') -Headers $headers -TimeoutSec 15
    if($main.sha -cne $SourceCommit){throw 'Public main differs from reviewed publication source'}
    $ciRuns=Invoke-RestMethod -Uri ($api+'/actions/runs?head_sha='+$SourceCommit+'&branch=main&per_page=100') -Headers $headers -TimeoutSec 15
    $green=@($ciRuns.workflow_runs | Where-Object {$_.head_sha -ceq $SourceCommit -and $_.head_branch -ceq 'main' -and $_.name -ceq 'KernelBench-RL CI' -and $_.status -ceq 'completed' -and $_.conclusion -ceq 'success'})
    if(-not $green.Count){throw 'Exact-source main CI is not successfully complete'}
    $receipt.ci_url=$green[0].html_url
    $handler=[Net.Http.HttpClientHandler]::new()
    $handler.AutomaticDecompression=[Net.DecompressionMethods]::GZip -bor [Net.DecompressionMethods]::Deflate
    # This public client deliberately has no GitHub Authorization header.
    $client=[Net.Http.HttpClient]::new($handler);$client.Timeout=[TimeSpan]::FromSeconds(15)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd('KernelBench-public-byte-verifier')
    $client.DefaultRequestHeaders.CacheControl=[Net.Http.Headers.CacheControlHeaderValue]::new()
    $client.DefaultRequestHeaders.CacheControl.NoCache=$true
    $deadline=[DateTime]::UtcNow.AddSeconds($MaxWaitSeconds)
    do{
        $attempt=[ordered]@{observed_utc=[DateTime]::UtcNow.ToString('o');verified=$false;files=@()}
        try{
            $pages=Invoke-RestMethod -Uri ($api+'/pages') -Headers $headers -TimeoutSec 15
            if($pages.cname -or $pages.source.branch -cne 'main' -or $pages.source.path -cne '/docs'){throw 'Pages configuration differs from main/docs'}
            $build=Invoke-RestMethod -Uri ($api+'/pages/builds/latest') -Headers $headers -TimeoutSec 15
            $attempt.build_status=$build.status;$attempt.build_commit=$build.commit
            if($build.status -cne 'built' -or $build.commit -cne $SourceCommit){throw 'Exact-source Pages build is pending or failed'}
            foreach($name in $names){
                if($name -ceq 'docs/.nojekyll'){continue}
                $relative=$name.Substring(5)
                $url=$base+'/'+(($relative -split '/' | ForEach-Object {[uri]::EscapeDataString($_)}) -join '/')
                $response=$client.GetAsync($url).GetAwaiter().GetResult()
                try{
                    $response.EnsureSuccessStatusCode() | Out-Null
                    if($relative -ceq $pdf -and $response.Content.Headers.ContentType.MediaType -cne 'application/pdf'){throw 'Public paper lacks PDF content type'}
                    $finalUri=$response.RequestMessage.RequestUri.AbsoluteUri
                    if(-not $finalUri.StartsWith($base+'/',[StringComparison]::Ordinal)){throw 'Public asset redirected away from project origin'}
                    $bytes=$response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
                    $digest=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
                    $local=Get-Item -LiteralPath (Join-Path $RepoRoot $name)
                    $expected=(Get-FileHash -LiteralPath $local.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                    if($bytes.Length -ne $local.Length -or $digest -cne $expected){throw ('Public/source bytes differ: '+$relative)}
                    $attempt.files+=@{path=$relative;url=$url;bytes=$bytes.Length;sha256=$digest}
                }finally{$response.Dispose()}
            }
            $attempt.verified=$true;$receipt.verified=$true;$receipt.publication_complete=$true
            $receipt.paper_pdf_url=$base+'/'+$pdf;$receipt.metadata_robots_sitemap_verified=$true
            $receipt.completed_utc=[DateTime]::UtcNow.ToString('o')
        }catch{$attempt.error=$_.Exception.Message}
        $receipt.attempts+=@($attempt)
        $json=$receipt | ConvertTo-Json -Depth 10
        [IO.File]::WriteAllText([IO.Path]::GetFullPath($ReceiptPath),$json+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
        if($receipt.verified -or [DateTime]::UtcNow -ge $deadline){break}
        Start-Sleep -Seconds ([Math]::Min($PollSeconds,[Math]::Max(1,[int]($deadline-[DateTime]::UtcNow).TotalSeconds)))
    }while([DateTime]::UtcNow -lt $deadline)
    if($receipt.verified){Write-Output ('PAGES_VERIFIED '+$base)}else{Write-Output ('PAGES_PENDING '+$base+'; receipt retained; rerun verification to resume')}
}finally{if($client){$client.Dispose()};$credentialMap.Clear()}
