# Local test script for Barrier Forge Patched Test
# Run this from the quantlib-forge-company directory
# This demonstrates successful kernel reuse WITH Forge-aware patches

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Local Barrier Forge Test (Patched)" -ForegroundColor Cyan
Write-Host "  Expected to PASS - demonstrates kernel reuse" -ForegroundColor Green
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

echo "=== Applying QuantLib patches ==="
cd quantlib
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

echo "=== Switching test file to ABool version ==="
sed -i 's/barrieroption_forge\.cpp/barrieroption_forge_abool.cpp/' ../quantlib-forge/forge-test-suite/CMakeLists.txt

echo "=== Building QuantLib (Patched) ==="
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/workspace/install-release -DCMAKE_CXX_FLAGS="-Wno-reorder" -DQL_BUILD_TEST_SUITE=OFF -DQL_BUILD_EXAMPLES=OFF -DQL_BUILD_BENCHMARK=OFF -DQL_ENABLE_SESSIONS=OFF -DQL_NULL_AS_FUNCTIONS=ON -DQL_FORGE_DISABLE_AAD=OFF -DQL_FORGE_BUILD_TEST_SUITE=ON -DQL_EXTERNAL_SUBDIRECTORIES="../quantlib-forge" -DQL_EXTRA_LINK_LIBRARIES="QuantLib-Forge"
cmake --build build --config Release

echo ""
echo "============================================================="
echo "  BARRIER FORGE TEST - PATCHED"
echo "============================================================="
echo ""
echo "This test demonstrates successful kernel reuse:"
echo "  - Kernel is compiled with Input Set 1 (strike=100 > barrier)"
echo "  - Re-evaluated with Input Set 5 (strike=80 < barrier=95)"
echo "  - With Forge-aware patches, ABool::If records all branches"
echo "  - Expected: ALL input sets PASS"
echo ""
echo "Patches applied:"
echo "  - ErrorFunction: ABool::If for 4 branch regions"
echo "  - CumulativeNormalDistribution: ABool::If for asymptotic branch"
echo "  - AnalyticBarrierEngine: ABool::If for strike/barrier conditions"
echo ""
echo "============================================================="
echo ""

TEST_EXE=$(find ./build -name "forge-test-suite" -type f -executable | head -1)
if [ -n "$TEST_EXE" ]; then
    # Run the test - all should pass
    $TEST_EXE --log_level=message --run_test=QuantLibForgeRisksTests/BarrierOptionForgeTest/testForgeBarrierKernelReuse
    
    echo ""
    echo "============================================================="
    echo "  SUCCESS: All input sets passed!"
    echo "  Kernel reuse works correctly with Forge-aware patches."
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

