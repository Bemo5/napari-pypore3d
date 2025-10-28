# PyPore3d-Installation-Script.ps1  (Targets Python 3.12, uses existing SWIG)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$PyTag = '-3.12'

function Write-Section($msg){ Write-Host "`n==== $msg ====" -ForegroundColor Cyan }

function Backup-EnvironmentVariables {
  try {
    $backupDir = "C:\Users\$Env:UserName\Desktop\Environment-Variable-Backup"
    if (-not (Test-Path $backupDir)) { New-Item -ItemType Directory -Path $backupDir | Out-Null }
    $machinePath = [Environment]::GetEnvironmentVariable('Path','Machine')
    $userPath    = [Environment]::GetEnvironmentVariable('Path','User')
    $swigLibUsr  = [Environment]::GetEnvironmentVariable('SWIG_LIB','User')
    $swigLibMch  = [Environment]::GetEnvironmentVariable('SWIG_LIB','Machine')
    $machinePath | Out-File "$backupDir\machine-Path.txt" -Encoding UTF8
    $userPath    | Out-File "$backupDir\user-Path.txt"    -Encoding UTF8
    $swigLibUsr  | Out-File "$backupDir\user-SWIG_LIB.txt"    -Encoding UTF8
    $swigLibMch  | Out-File "$backupDir\machine-SWIG_LIB.txt" -Encoding UTF8
  } catch { Write-Warning "Backup failed: $($_.Exception.Message)" }
}

function Ensure-Admin {
  $IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).
             IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
  if (-not $IsAdmin) {
    Start-Process -FilePath "powershell.exe" -ArgumentList "-File `"$PSCommandPath`"" -Verb RunAs
    exit
  }
}

function Ensure-Python312 {
  Write-Section "Checking Python 3.12"
  $has3_12 = $false
  if (Get-Command py -ErrorAction SilentlyContinue) {
    $list = (& py -0p) -join "`n"
    if ($list -match '3\.12') { $has3_12 = $true }
  }
  if (-not $has3_12) {
    Write-Host "Installing Python 3.12 via winget..."
    try {
      winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    } catch {
      Write-Host "winget failed; falling back to python.org installer..."
      $pyUrl = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
      $pyExe = "$env:TEMP\python-3.12.0-amd64.exe"
      Invoke-WebRequest $pyUrl -OutFile $pyExe
      Start-Process -FilePath $pyExe -ArgumentList "/quiet PrependPath=1 Include_launcher=1" -Wait
    }
  }
  & py $PyTag -V | Write-Host
}

function Ensure-BuildTools {
  Write-Section "Checking MSVC Build Tools"
  $clOk = $false
  try { & cmd /c cl 2>&1 | Out-Null; $clOk = $true } catch { $clOk = $false }
  if (-not $clOk) {
    Write-Host "Installing Visual Studio 2022 Build Tools (VC Tools)..."
    $bt = "$env:TEMP\vs_buildtools.exe"
    Invoke-WebRequest "https://aka.ms/vs/17/release/vs_BuildTools.exe" -OutFile $bt
    Start-Process -FilePath $bt -ArgumentList `
      "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended" -Wait
  }
}

function Ensure-SWIG {
  Write-Section "Checking SWIG"
  if (-not (Get-Command swig -ErrorAction SilentlyContinue)) {
    throw "SWIG is not on PATH. Since you already installed it manually, please ensure swig.exe is reachable (e.g., C:\Tools\swigwin-4.3.1)."
  }

  # Make sure SWIG_LIB is visible to child processes (build steps)
  $swigLib = $env:SWIG_LIB
  if (-not $swigLib -or -not (Test-Path $swigLib)) {
    # derive from swig -swiglib
    $detected = (& swig -swiglib).Trim()
    if (-not (Test-Path $detected)) {
      # last-resort guess from common location
      if (Test-Path 'C:\Tools') {
        $rootGuess = (Get-ChildItem 'C:\Tools' -Filter 'swigwin-*' -Directory | Select-Object -First 1).FullName
        $detected = Join-Path $rootGuess 'Lib'
      }
    }
    if (-not (Test-Path $detected)) { throw "Could not resolve SWIG_LIB; set it to <swigroot>\Lib and re-run." }

    # Set for current process + User (persist)
    $env:SWIG_LIB = $detected
    [Environment]::SetEnvironmentVariable('SWIG_LIB', $detected, 'User')
  }

  Write-Host "SWIG: $((swig -version) -join ' ')"
  Write-Host "SWIG_LIB: $env:SWIG_LIB"
}

function Py312 {
  param([Parameter(Mandatory=$true)][string]$ArgsLine)
  & py $PyTag $ArgsLine
}

function Ensure-PipAndDeps {
  Write-Section "Upgrading pip/setuptools/wheel and installing deps"
  Py312 "-m ensurepip --upgrade"
  Py312 "-m pip install -U pip setuptools wheel"
  Py312 "-m pip install -U numpy"
  Py312 "-m pip install -U SimpleITK"
}

function Install-PyPore3D {
  Write-Section "Installing PyPore3D"

  # Try PyPI first (cleanest). If it fails, fall back to the Windows zip.
  $pypiOk = $true
  try {
    Py312 "-m pip install --no-cache-dir pypore3d"
  } catch {
    $pypiOk = $false
    Write-Warning "PyPI install failed: $($_.Exception.Message)"
  }

  if (-not $pypiOk) {
    Write-Host "Falling back to the Windows ZIP package..."
    $url = "https://gitlab.elettra.eu/aboulhassan.amal/PyPore3D/-/wikis/uploads/22d2cb768bc4934a88685f80810470f8/PyPore3D_Win.zip"
    $zip = "$env:TEMP\PyPore3D_Win.zip"
    Invoke-WebRequest $url -OutFile $zip
    if ((Get-Item $zip).Length -lt 100KB) { throw "PyPore3D_Win.zip looks invalid (too small)." }

    $dest = "C:\Program Files\PyPore3D_Win"
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Expand-Archive -Path $zip -DestinationPath "C:\Program Files" -Force

    Push-Location $dest
    try {
      # Prefer pip so it installs correctly into site-packages
      Py312 "-m pip install --no-cache-dir --no-build-isolation ."
    } catch {
      Write-Warning "pip install (from ZIP) failed: $($_.Exception.Message). Trying legacy build_ext..."
      Py312 "setup.py build_ext --inplace"
      # manual copy as last resort
      $site = Py312 "-c `"import site,sys; print([p for p in site.getsitepackages() if 'site-packages' in p][0])`""
      $site = $site.Trim()
      if (-not (Test-Path $site)) { throw "Could not resolve site-packages path." }
      if (Test-Path (Join-Path $site 'pypore3d')) { Remove-Item (Join-Path $site 'pypore3d') -Recurse -Force }
      Copy-Item -Recurse -Force (Join-Path $PWD 'pypore3d') -Destination $site
    } finally {
      Pop-Location
    }
  }

  Write-Section "Verifying import"
  Py312 "-c `"import pypore3d,sys; print('PyPore3D OK on', sys.version.split()[0], '->', pypore3d.__version__)`""
}

# ---------------- MAIN ----------------
Ensure-Admin
Backup-EnvironmentVariables
Ensure-Python312
Ensure-BuildTools
Ensure-SWIG
Ensure-PipAndDeps
Install-PyPore3D

Write-Host "`nAll done."
