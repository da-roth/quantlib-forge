# Local benchmark script for XAD
# Run this from the quantlib-forge-company directory
# Builds QuantLib with XAD (QuantLib-Risks-Cpp) and runs the XVA benchmark

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Local XAD Benchmark" -ForegroundColor Cyan
Write-Host "  QuantLib with XAD (tape-based AAD)" -ForegroundColor Cyan
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

echo "=== Cloning XAD ==="
git clone https://github.com/auto-differentiation/xad.git xad

echo "=== Cloning QuantLib-Risks-Cpp ==="
git clone https://github.com/auto-differentiation/QuantLib-Risks-Cpp.git QuantLib-Risks-Cpp

echo "=== Configuring QuantLib with XAD ==="
cd QuantLib
mkdir build
cd build
cmake -G Ninja -DBOOST_ROOT=/usr \
  -DQLRISKS_DISABLE_AAD=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DQL_EXTERNAL_SUBDIRECTORIES="$(pwd)/../../xad;$(pwd)/../../QuantLib-Risks-Cpp" \
  -DQL_EXTRA_LINK_LIBRARIES=QuantLib-Risks \
  -DQL_NULL_AS_FUNCTIONS=ON \
  ..

echo "=== Compiling QuantLib ==="
cmake --build .

echo "=== Installing QuantLib ==="
cmake --install . --prefix /workspace/install

echo "=== Building XAD Benchmark ==="
cd /workspace
mkdir benchmark-build
cd benchmark-build

cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.15)
project(SwapXvaXad LANGUAGES CXX)
find_package(QuantLib-Risks REQUIRED)
add_executable(swap_xva_xad /workspace/quantlib-forge/benchmarks/swap_xva_xad.cpp)
target_link_libraries(swap_xva_xad PRIVATE QuantLib::QuantLib)
target_compile_features(swap_xva_xad PUBLIC cxx_std_17)
EOF

cmake -G Ninja -DBOOST_ROOT=/usr \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/workspace/install \
  .
cmake --build .

echo ""
echo "============================================================="
echo "  XAD Benchmark - QuantLib with XAD (tape-based AAD)"
echo "============================================================="
./swap_xva_xad
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

