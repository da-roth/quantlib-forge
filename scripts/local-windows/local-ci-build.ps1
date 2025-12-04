# Local CI Build script (Native Windows)
# Run this from the quantlib-forge-company directory
# Builds QuantLib with Forge (no tests, no patches) - fast compilation check

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Local CI Build (Native Windows)" -ForegroundColor Cyan
Write-Host "  Build only - no tests, no patches" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're on Windows
if ($PSVersionTable.Platform -and $PSVersionTable.Platform -ne "Win32NT") {
    Write-Host "ERROR: This script is designed to run on Windows only." -ForegroundColor Red
    exit 1
}

$BOOST_VERSION = "1_84_0"
$BOOST_VERSION_DOT = "1.84.0"
$BOOST_ROOT = "C:/local/boost_$BOOST_VERSION"

# Check if Boost is already installed
if (-not (Test-Path $BOOST_ROOT)) {
    Write-Host "=== Installing Boost ===" -ForegroundColor Yellow

    $boostUrl = "https://archives.boost.io/release/$BOOST_VERSION_DOT/source/boost_$BOOST_VERSION.zip"
    $boostZip = "C:/temp/boost.zip"
    $boostExtract = "C:/temp/boost"

    New-Item -ItemType Directory -Force -Path C:/temp | Out-Null
    New-Item -ItemType Directory -Force -Path C:/local | Out-Null

    Write-Host "Downloading Boost..."
    Invoke-WebRequest -Uri $boostUrl -OutFile $boostZip

    Write-Host "Extracting Boost..."
    Expand-Archive -Path $boostZip -DestinationPath $boostExtract -Force

    Move-Item -Path "$boostExtract/boost_$BOOST_VERSION" -Destination $BOOST_ROOT -Force

    Write-Host "Building Boost..."
    Push-Location $BOOST_ROOT
    & ./bootstrap.bat
    & ./b2 --with-test --with-system --with-filesystem --with-serialization --with-regex --with-date_time link=static runtime-link=static variant=release address-model=64 threading=multi
    Pop-Location

    Write-Host "Boost installation complete." -ForegroundColor Green
} else {
    Write-Host "Boost already installed at $BOOST_ROOT" -ForegroundColor Green
}

# Clone repositories if needed
if (-not (Test-Path "forge")) {
    Write-Host "=== Cloning Forge ===" -ForegroundColor Yellow
    git clone https://github.com/da-roth/forge.git
}

if (-not (Test-Path "quantlib")) {
    Write-Host "=== Cloning QuantLib ===" -ForegroundColor Yellow
    git clone https://github.com/lballabio/QuantLib.git quantlib
    Push-Location quantlib
    git checkout b3612ef
    Pop-Location
}

# Build Forge
Write-Host "=== Building Forge ===" -ForegroundColor Yellow
Push-Location forge
$installPrefix = Join-Path $PWD ".." "install-release"
cmake -B build -S tools/packaging `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX=$installPrefix
cmake --build build --config Release
cmake --install build --config Release
Pop-Location

# Build QuantLib with QuantLib-Forge (no tests, no patches)
Write-Host "=== Building QuantLib with QuantLib-Forge (no tests, no patches) ===" -ForegroundColor Yellow
Push-Location quantlib
$workspacePath = (Get-Location).Parent.FullName
cmake -B build -S . `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH=$installPrefix `
    -DBOOST_ROOT=$BOOST_ROOT `
    -DBoost_USE_STATIC_LIBS=ON `
    -DBoost_USE_STATIC_RUNTIME=ON `
    -DQL_BUILD_TEST_SUITE=OFF `
    -DQL_BUILD_EXAMPLES=OFF `
    -DQL_BUILD_BENCHMARK=OFF `
    -DQL_ENABLE_SESSIONS=OFF `
    -DQL_NULL_AS_FUNCTIONS=ON `
    -DQL_FORGE_DISABLE_AAD=OFF `
    -DQL_FORGE_BUILD_TEST_SUITE=OFF `
    -DQL_EXTERNAL_SUBDIRECTORIES="$workspacePath" `
    -DQL_EXTRA_LINK_LIBRARIES="QuantLib-Forge"
cmake --build build --config Release
Pop-Location

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Build Successful (Windows)" -ForegroundColor Cyan
Write-Host "  QuantLib-Forge compiled without errors" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
