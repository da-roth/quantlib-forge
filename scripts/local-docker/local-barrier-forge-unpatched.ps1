# Local test script for Barrier Forge Unpatched Test
# Run this from the quantlib-forge-company directory
# This demonstrates the kernel reuse problem WITHOUT Forge-aware patches

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Local Barrier Forge Test (Unpatched)" -ForegroundColor Cyan
Write-Host "  Expected to FAIL - demonstrates kernel reuse problem" -ForegroundColor Yellow
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

echo "=== Applying Forge algebraic simplification fix ==="
# algebraic_simplification.cpp patch is now in Forge main branch, no longer needed

echo "=== Building Forge ==="
cd forge
cmake -B build -S tools/packaging -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/workspace/install-release
cmake --build build --config Release
cmake --install build --config Release
cd /workspace

echo "=== Applying QuantLib ABool patch (base types only, NO Forge-aware functions) ==="
cd quantlib
sed 's/\r$//' ../quantlib-forge/patches/quantlib/quantlib-abool.patch | git apply -
echo ""
echo "NOTE: Only applying base ABool types patch."
echo "NOT applying Forge-aware ErrorFunction/CDF/BarrierEngine patches."
echo "This means conditional branches use C++ if/else, not ABool::If."
echo ""

echo "=== Switching test file to ABool version ==="
sed -i 's/barrieroption_forge\.cpp/barrieroption_forge_abool.cpp/' ../quantlib-forge/forge-test-suite/CMakeLists.txt

echo "=== Building QuantLib (Unpatched) ==="
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/workspace/install-release -DCMAKE_CXX_FLAGS="-Wno-reorder" -DQL_BUILD_TEST_SUITE=OFF -DQL_BUILD_EXAMPLES=OFF -DQL_BUILD_BENCHMARK=OFF -DQL_ENABLE_SESSIONS=OFF -DQL_NULL_AS_FUNCTIONS=ON -DQL_FORGE_DISABLE_AAD=OFF -DQL_FORGE_BUILD_TEST_SUITE=ON -DQL_EXTERNAL_SUBDIRECTORIES="../quantlib-forge" -DQL_EXTRA_LINK_LIBRARIES="QuantLib-Forge"
cmake --build build --config Release

echo ""
echo "============================================================="
echo "  BARRIER FORGE TEST - UNPATCHED"
echo "============================================================="
echo ""
echo "This test demonstrates the kernel reuse problem:"
echo "  - Kernel is compiled with Input Set 1 (strike=100 > barrier)"
echo "  - Re-evaluated with Input Set 5 (strike=80 < barrier=95)"
echo "  - Without Forge-aware patches, branch conditions are baked in"
echo "  - Expected: Input Set 5 FAILS with large error (~144%)"
echo ""
echo "============================================================="
echo ""

TEST_EXE=$(find ./build -name "forge-test-suite" -type f -executable | head -1)
if [ -n "$TEST_EXE" ]; then
    # Run the test - we expect it to fail on Input Set 5
    # Use || true to not fail the script, we want to show the output
    $TEST_EXE --log_level=message --run_test=QuantLibForgeRisksTests/BarrierOptionForgeTest/testForgeBarrierKernelReuse 2>&1 || true
    
    echo ""
    echo "============================================================="
    echo "  EXPECTED RESULT: Input Set 5 should FAIL"
    echo "  This demonstrates why Forge-aware patches are needed."
    echo "============================================================="
else
    echo "Test executable not found!"
    exit 1
fi
'@

$tempScript = Join-Path $env:TEMP "run-test.sh"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$unixContent = $containerScript -replace "`r`n", "`n" -replace "`r", "`n"
[System.IO.File]::WriteAllText($tempScript, $unixContent, $utf8NoBom)

Write-Host "Running Docker container..." -ForegroundColor Yellow
Write-Host ""

docker run --rm -v "${PWD}:/workspace/quantlib-forge" -v "${tempScript}:/workspace/run-test.sh" ghcr.io/lballabio/quantlib-devenv:rolling bash /workspace/run-test.sh

Remove-Item $tempScript -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Test Complete" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

