# Build SppToPawPaint.exe and the release zip.
#   powershell -ExecutionPolicy Bypass -File build.ps1
# Needs: Python 3.10+ with  pip install -r converter/requirements.txt pyinstaller
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$version = (Select-String -Path "$root\converter\spp2pawpaint\__init__.py" -Pattern '__version__ = "(.+)"').Matches[0].Groups[1].Value

Push-Location "$root\converter"
try {
    python -m PyInstaller --noconfirm --onefile --console --name SppToPawPaint `
        --distpath dist --workpath "$env:TEMP\spp2pawpaint_build" --specpath "$env:TEMP\spp2pawpaint_build" `
        --collect-submodules h5py `
        --collect-all moderngl --collect-all glcontext --collect-submodules numexpr `
        --add-data "$root\plugin\scripts;plugin_scripts" `
        --exclude-module matplotlib --exclude-module scipy --exclude-module pandas `
        --exclude-module zstandard --exclude-module xxhash --exclude-module pytest --exclude-module lupa `
        SppToPawPaint.py
} finally { Pop-Location }

$rel = "$root\release\SppToPawPaint-$version"
if (Test-Path $rel) { Remove-Item -Recurse -Force $rel }
New-Item -ItemType Directory -Force "$rel\PawPaint scripts", "$rel\Substance plugin (optional)" | Out-Null
Copy-Item "$root\converter\dist\SppToPawPaint.exe" $rel
Copy-Item "$root\plugin\scripts\spp_importer.lua", "$root\plugin\scripts\autoload.lua" "$rel\PawPaint scripts"
Copy-Item "$root\substance_plugin\pawpaint_exporter.py", "$root\substance_plugin\README.md" "$rel\Substance plugin (optional)"
Copy-Item "$root\release_readme.txt" "$rel\READ ME FIRST.txt"
Copy-Item "$root\LICENSE", "$root\THIRD_PARTY_NOTICES.txt" $rel
$zip = "$root\release\SppToPawPaint-$version.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $rel -DestinationPath $zip
Get-FileHash $zip -Algorithm SHA256 | ForEach-Object { "$($_.Hash)  SppToPawPaint-$version.zip" } |
    Set-Content "$root\release\SppToPawPaint-$version.zip.sha256"
Write-Host "Built $zip"
