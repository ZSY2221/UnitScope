param(
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Output) { $Output = Join-Path $root 'data\ncvr' }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$items = @(
    [pscustomobject]@{
        name = 'ncvoter_2026-07-13.zip'
        version = 'hhBFefdjI7raPYd2N7ba5mKubUv_D346'
        last_modified = '2026-07-13T00:21:24Z'
        size = 517980796
    }
)
foreach ($item in $items) {
    $path = Join-Path $Output $item.name
    $url = 'https://s3.amazonaws.com/dl.ncsbe.gov/data/ncvoter_Statewide.zip'
    if ((Test-Path -LiteralPath $path) -and (Get-Item -LiteralPath $path).Length -eq $item.size) {
        Write-Host "SKIP $($item.name)"
        continue
    }
    Write-Host "DOWNLOAD $($item.name)"
    & curl.exe -L --fail --retry 8 --retry-delay 5 --continue-at - --output $path $url
    if ($LASTEXITCODE -ne 0) { throw "NCVR download failed for $($item.name)" }
    if ((Get-Item -LiteralPath $path).Length -ne $item.size) {
        throw "NCVR size mismatch for $($item.name)"
    }
}
$provenance = [ordered]@{
    source = 'North Carolina State Board of Elections public voter registration file'
    object = 'data/ncvoter_Statewide.zip'
    source_url = 'https://s3.amazonaws.com/dl.ncsbe.gov/data/ncvoter_Statewide.zip'
    versions = $items
    use = 'field distributions for entity-resolution and deletion-reorganization experiments'
    view = 'two deterministic views derived from one official snapshot'
    publication = 'aggregate results only'
}
$provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Output 'provenance.json') -Encoding UTF8
Write-Host "NCVR downloads complete: $Output"
