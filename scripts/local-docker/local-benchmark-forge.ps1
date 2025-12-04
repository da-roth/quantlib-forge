# Local Forge Benchmark script
# Run this from the quantlib-forge-company directory
# Builds QuantLib with Forge and runs the comprehensive XVA benchmark
# comparing ALL approaches: Bump-Reval, Forward-Only, and AAD

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Local Forge Benchmark" -ForegroundColor Cyan
Write-Host "  All Approaches: Bump-Reval, Forward-Only, AAD" -ForegroundColor Cyan
Write-Host "  SSE2 vs AVX2, Stability vs AllOpt vs NoOpt" -ForegroundColor Cyan
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
git clone https://github.com/lballabio/QuantLib.git QuantLib
cd QuantLib
git checkout b3612ef
cd /workspace

echo "=== Checking CPU capabilities ==="
echo "============================================================="
echo "  CPU Capabilities"
echo "============================================================="
cat /proc/cpuinfo | grep -m1 "model name" || true
echo ""
echo "AVX2 support:"
grep -q avx2 /proc/cpuinfo && echo "  AVX2: SUPPORTED" || echo "  AVX2: NOT SUPPORTED"
grep -q avx /proc/cpuinfo && echo "  AVX:  SUPPORTED" || echo "  AVX:  NOT SUPPORTED"
grep -q sse2 /proc/cpuinfo && echo "  SSE2: SUPPORTED" || echo "  SSE2: NOT SUPPORTED"

echo "=== Building Forge ==="
cd forge
cmake -B build -S tools/packaging \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/workspace/install
cmake --build build --config Release
cmake --install build --config Release

echo "=== Configuring QuantLib with QuantLib-Forge ==="
cd /workspace/QuantLib
mkdir build
cd build
cmake -G Ninja -DBOOST_ROOT=/usr \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/workspace/install \
  -DCMAKE_CXX_FLAGS="-Wno-reorder" \
  -DQL_BUILD_TEST_SUITE=OFF \
  -DQL_BUILD_EXAMPLES=OFF \
  -DQL_BUILD_BENCHMARK=OFF \
  -DQL_ENABLE_SESSIONS=ON \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DQL_FORGE_DISABLE_AAD=OFF \
  -DQL_FORGE_BUILD_TEST_SUITE=OFF \
  -DQL_EXTERNAL_SUBDIRECTORIES="../quantlib-forge" \
  -DQL_EXTRA_LINK_LIBRARIES="QuantLib-Forge" \
  ..

echo "=== Compiling QuantLib ==="
cmake --build .

echo "=== Installing QuantLib ==="
cmake --install . --prefix /workspace/install

echo "=== Building Comprehensive Forge Benchmark ==="
cd /workspace
mkdir benchmark-build
cd benchmark-build

cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.15)
project(SwapXvaForgeAll LANGUAGES CXX)
find_package(QuantLib REQUIRED)
find_package(Forge CONFIG REQUIRED)
add_executable(swap_xva_forge_all /workspace/quantlib-forge/benchmarks/swap_xva_forge_all.cpp)
target_link_libraries(swap_xva_forge_all PRIVATE QuantLib::QuantLib Forge::forge)
target_compile_features(swap_xva_forge_all PUBLIC cxx_std_17)
target_compile_options(swap_xva_forge_all PRIVATE -mavx2)
EOF

cmake -G Ninja -DBOOST_ROOT=/usr \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/workspace/install \
  .
cmake --build .

echo ""
echo "============================================================="
echo "  Forge Benchmark - All Approaches"
echo "============================================================="
./swap_xva_forge_all
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

Write-Host ""


