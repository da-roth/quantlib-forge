# Local CI Build script (Docker/Linux)
# Run this from the quantlib-forge-company directory
# Builds QuantLib with Forge (no tests, no patches) - fast compilation check

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Local CI Build (Docker/Linux)" -ForegroundColor Cyan
Write-Host "  Build only - no tests, no patches" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

$containerScript = @'
#!/bin/bash
set -e

cd /workspace

echo "=== Installing ninja ==="
apt-get update && apt-get install -y ninja-build

echo "=== Cloning Forge ==="
git clone https://github.com/da-roth/forge.git

echo "=== Cloning QuantLib ==="
git clone https://github.com/lballabio/QuantLib.git quantlib
cd quantlib
git checkout b3612ef
cd /workspace

echo "=== Building Forge ==="
cd forge
cmake -B build -S tools/packaging \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/workspace/install-release
cmake --build build --config Release
cmake --install build --config Release

echo "=== Building QuantLib with QuantLib-Forge (no tests, no patches) ==="
cd /workspace/quantlib
cmake -B build -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/workspace/install-release \
  -DCMAKE_CXX_FLAGS="-Wno-reorder" \
  -DQL_BUILD_TEST_SUITE=OFF \
  -DQL_BUILD_EXAMPLES=OFF \
  -DQL_BUILD_BENCHMARK=OFF \
  -DQL_ENABLE_SESSIONS=OFF \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DQL_FORGE_DISABLE_AAD=OFF \
  -DQL_FORGE_BUILD_TEST_SUITE=OFF \
  -DQL_EXTERNAL_SUBDIRECTORIES="../quantlib-forge" \
  -DQL_EXTRA_LINK_LIBRARIES="QuantLib-Forge"
cmake --build build --config Release

echo ""
echo "============================================================="
echo "  Build Successful (Linux/Docker)"
echo "  QuantLib-Forge compiled without errors"
echo "============================================================="
'@

$tempScript = Join-Path $env:TEMP "run-build.sh"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$unixContent = $containerScript -replace "`r`n", "`n" -replace "`r", "`n"
[System.IO.File]::WriteAllText($tempScript, $unixContent, $utf8NoBom)

Write-Host "Running Docker container..." -ForegroundColor Yellow
Write-Host ""

docker run --rm -v "${PWD}:/workspace/quantlib-forge" -v "${tempScript}:/workspace/run-build.sh" ghcr.io/lballabio/quantlib-devenv:rolling bash /workspace/run-build.sh

Remove-Item $tempScript -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  CI Build Complete" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
