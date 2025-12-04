# Local test script for Minimal CDF Kernel Reuse Test
# Run this from the quantlib-forge-company directory
# This tests the CDF implementation directly without building QuantLib

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Local Minimal CDF Kernel Reuse Test" -ForegroundColor Cyan
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

echo "=== Applying Forge algebraic simplification fix ==="
# algebraic_simplification.cpp patch is now in Forge main branch, no longer needed

echo "=== Building Forge ==="
cd forge
cmake -B build -S tools/packaging -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/workspace/install-forge
cmake --build build --config Release
cmake --install build --config Release
cd /workspace

echo "=== Building Minimal CDF Test ==="
mkdir -p test-build
cd test-build

cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.15)
project(MinimalCDFTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Forge package
find_package(Forge CONFIG REQUIRED)

# Add the expressions library from quantlib-forge
add_subdirectory(/workspace/quantlib-forge/expressions expressions)

add_executable(minimal_cdf_test
  /workspace/quantlib-forge/tests/minimal_cdf_test.cpp
)

target_include_directories(minimal_cdf_test PRIVATE
  /workspace/quantlib-forge/tests
)

target_link_libraries(minimal_cdf_test PRIVATE forge-expressions)
EOF

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/workspace/install-forge .
cmake --build .

echo ""
echo "============================================="
echo "  Running Minimal CDF Kernel Reuse Test"
echo "============================================="
echo ""
./minimal_cdf_test
'@

$tempScript = Join-Path $env:TEMP "run-test.sh"
# Write with Unix line endings (LF only, no BOM)
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
