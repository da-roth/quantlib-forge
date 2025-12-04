# Local CI Test + QuantLib Tests script (Docker/Linux)
# Run this from the quantlib-forge-company directory
# Builds QuantLib with Forge with all patches applied and runs BOTH test suites

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Local CI Test + QuantLib Tests (Docker/Linux)" -ForegroundColor Cyan
Write-Host "  Full build with patches and ALL tests" -ForegroundColor Cyan
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

echo "=== Applying QuantLib patches ==="
cd /workspace/quantlib
echo "Applying QuantLib ABool patch..."
sed 's/\r$//' ../quantlib-forge/patches/quantlib/quantlib-abool.patch | git apply -
echo "Applying Forge-aware ErrorFunction patches..."
sed 's/\r$//' ../quantlib-forge/patches/quantlib/quantlib-errorfunction.hpp.patch | git apply -
sed 's/\r$//' ../quantlib-forge/patches/quantlib/quantlib-errorfunction.cpp.patch | git apply -
echo "Applying Forge-aware NormalDistribution patches..."
sed 's/\r$//' ../quantlib-forge/patches/quantlib/quantlib-normaldistribution.hpp.patch | git apply -
sed 's/\r$//' ../quantlib-forge/patches/quantlib/quantlib-normaldistribution.cpp.patch | git apply -
echo "Applying Forge-aware AnalyticBarrierEngine patch..."
sed 's/\r$//' ../quantlib-forge/patches/quantlib/quantlib-analyticbarrierengine.cpp.patch | git apply -
echo "All QuantLib patches applied."

echo "=== Building QuantLib with QuantLib-Forge (with ALL tests) ==="
cmake -B build -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/workspace/install-release \
  -DCMAKE_CXX_FLAGS="-Wno-reorder" \
  -DQL_BUILD_TEST_SUITE=ON \
  -DQL_BUILD_EXAMPLES=OFF \
  -DQL_BUILD_BENCHMARK=OFF \
  -DQL_ENABLE_SESSIONS=ON \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DQL_FORGE_DISABLE_AAD=OFF \
  -DQL_FORGE_BUILD_TEST_SUITE=ON \
  -DQL_EXTERNAL_SUBDIRECTORIES="../quantlib-forge" \
  -DQL_EXTRA_LINK_LIBRARIES="QuantLib-Forge"
cmake --build build --config Release

echo ""
echo "============================================================="
echo "  Running QuantLib-Forge Tests (Linux/Docker)"
echo "============================================================="

FORGE_TEST_EXE=$(find ./build -name "forge-test-suite" -type f -executable | head -1)
if [ -n "$FORGE_TEST_EXE" ]; then
    $FORGE_TEST_EXE --log_level=message

    echo ""
    echo "============================================================="
    echo "  Forge Tests Passed (Linux/Docker)"
    echo "============================================================="
else
    echo "Forge test executable not found!"
    exit 1
fi

echo ""
echo "============================================================="
echo "  Running QuantLib Tests (Linux/Docker)"
echo "============================================================="

QL_TEST_EXE=$(find ./build -name "quantlib-test-suite" -type f -executable | head -1)
if [ -n "$QL_TEST_EXE" ]; then
    $QL_TEST_EXE --log_level=message

    echo ""
    echo "============================================================="
    echo "  QuantLib Tests Passed (Linux/Docker)"
    echo "============================================================="
else
    echo "QuantLib test executable not found!"
    exit 1
fi
'@

$tempScript = Join-Path $env:TEMP "run-test-quantlib.sh"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$unixContent = $containerScript -replace "`r`n", "`n" -replace "`r", "`n"
[System.IO.File]::WriteAllText($tempScript, $unixContent, $utf8NoBom)

Write-Host "Running Docker container..." -ForegroundColor Yellow
Write-Host ""

docker run --rm -v "${PWD}:/workspace/quantlib-forge" -v "${tempScript}:/workspace/run-test.sh" ghcr.io/lballabio/quantlib-devenv:rolling bash /workspace/run-test.sh

Remove-Item $tempScript -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  CI Test + QuantLib Tests Complete" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
