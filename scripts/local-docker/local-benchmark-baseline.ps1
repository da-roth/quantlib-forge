# Local benchmark script for Baseline
# Run this from the quantlib-forge-company directory
# Builds plain QuantLib (no AAD) and runs the XVA benchmark with finite differences

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Local Baseline Benchmark" -ForegroundColor Cyan
Write-Host "  Plain QuantLib (Finite Differences)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

$containerScript = @'
#!/bin/bash
set -e

cd /workspace

echo "=== Installing ninja ==="
apt-get update && apt-get install -y ninja-build

echo "=== Cloning QuantLib ==="
git clone https://github.com/lballabio/QuantLib.git QuantLib
cd QuantLib
git checkout b3612ef
cd /workspace

echo "=== Configuring QuantLib (plain, no AAD) ==="
cd QuantLib
mkdir build
cd build
cmake -G Ninja -DBOOST_ROOT=/usr \
  -DCMAKE_BUILD_TYPE=Release \
  -DQL_BUILD_TEST_SUITE=OFF \
  -DQL_BUILD_EXAMPLES=OFF \
  -DQL_BUILD_BENCHMARK=OFF \
  ..

echo "=== Compiling QuantLib ==="
cmake --build .

echo "=== Installing QuantLib ==="
cmake --install . --prefix /workspace/install

echo "=== Building Baseline Benchmark ==="
cd /workspace
mkdir benchmark-build
cd benchmark-build

cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.15)
project(SwapXvaBaseline LANGUAGES CXX)
find_package(QuantLib REQUIRED)
add_executable(swap_xva_baseline /workspace/quantlib-forge/benchmarks/swap_xva_baseline.cpp)
target_link_libraries(swap_xva_baseline PRIVATE QuantLib::QuantLib)
target_compile_features(swap_xva_baseline PUBLIC cxx_std_17)
EOF

cmake -G Ninja -DBOOST_ROOT=/usr \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/workspace/install \
  .
cmake --build .

echo ""
echo "============================================================="
echo "  Baseline Benchmark - Plain QuantLib (Finite Differences)"
echo "============================================================="
./swap_xva_baseline
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
Write-Host "  Benchmark Complete" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

