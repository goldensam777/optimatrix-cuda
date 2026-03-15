# run_tests.ps1 — Automatisation complete des tests optimatrix-cuda
#
# Usage :
#   .\run_tests.ps1
#   .\run_tests.ps1 -Arch sm_86    # RTX 30xx
#   .\run_tests.ps1 -Arch sm_89    # RTX 40xx
#
# Enregistre tout dans results/report_YYYYMMDD_HHMMSS.txt

param(
    [string]$Arch = "sm_75"   # MX450 / RTX 20xx / T4
)

# ── Configuration ─────────────────────────────────────────────────────

$NVCC   = "nvcc"
$CFLAGS = "-O2 -arch=$Arch -Iinclude"
$OBJDIR = "obj"
$RESDIR = "results"

$Timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$ReportFile = "$RESDIR\report_$Timestamp.txt"

# ── Etat global ───────────────────────────────────────────────────────

$TotalTests    = 0
$PassedTests   = 0
$FailedTests   = 0
$CompileErrors = 0
$LinkErrors    = 0
$Log = [System.Collections.Generic.List[string]]::new()

# ── Helpers ───────────────────────────────────────────────────────────

function Write-Log {
    param([string]$msg, [string]$color = "White")
    Write-Host $msg -ForegroundColor $color
    $Script:Log.Add($msg) | Out-Null
}

function Write-Sep    { Write-Log ("=" * 62) }
function Write-SubSep { Write-Log ("-" * 62) }

function Invoke-Compile {
    param([string]$label, [string]$cmd)
    Write-Log "  Compiling $label ..." "Cyan"
    $out = cmd /c "$cmd 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Log "  [COMPILE ERROR] $label" "Red"
        foreach ($l in $out) { Write-Log "    $l" "Red" }
        $Script:CompileErrors++
        return $false
    }
    return $true
}

function Invoke-Link {
    param([string]$label, [string]$cmd, [string]$exe)
    Write-Log "  Linking  $label ..." "Cyan"
    $out = cmd /c "$cmd 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Log "  [LINK ERROR] $label" "Red"
        foreach ($l in $out) { Write-Log "    $l" "Red" }
        $Script:LinkErrors++
        return $false
    }
    if (-not (Test-Path $exe)) {
        Write-Log "  [LINK ERROR] executable introuvable : $exe" "Red"
        $Script:LinkErrors++
        return $false
    }
    return $true
}

function Invoke-Test {
    param([string]$label, [string]$exe)
    Write-Log ""
    Write-Log "  >>> $label" "Yellow"
    Write-SubSep

    if (-not (Test-Path $exe)) {
        Write-Log "  [SKIP] executable introuvable : $exe" "DarkYellow"
        return
    }

    $sw  = [System.Diagnostics.Stopwatch]::StartNew()
    $out = cmd /c "$exe 2>&1"
    $sw.Stop()

    foreach ($line in $out) { Write-Log "  $line" }

    $oks   = ($out | Select-String "\bOK\b"   -AllMatches).Matches.Count
    $fails = ($out | Select-String "\bFAIL\b" -AllMatches).Matches.Count

    $Script:TotalTests  += $oks + $fails
    $Script:PassedTests += $oks
    $Script:FailedTests += $fails

    $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 3)
    $color   = if ($fails -eq 0) { "Green" } else { "Red" }
    Write-Log ""
    Write-Log "  Duree : ${elapsed}s   OK=$oks   FAIL=$fails" $color
}

# ── Init ──────────────────────────────────────────────────────────────

New-Item -ItemType Directory -Force -Path $OBJDIR | Out-Null
New-Item -ItemType Directory -Force -Path $RESDIR | Out-Null

Write-Sep
Write-Log "  OPTIMATRIX-CUDA — Rapport de tests" "Magenta"
Write-Log "  Date    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "  GPU cible : $Arch"
Write-Log "  Rapport : $ReportFile"
Write-Sep

# ── GPU info ──────────────────────────────────────────────────────────

Write-Log ""
Write-Log "  GPU INFO" "Cyan"
Write-SubSep
$gpuOut = cmd /c "nvidia-smi 2>&1"
foreach ($l in $gpuOut) { Write-Log "  $l" }

# ── Compilation des objets ────────────────────────────────────────────

Write-Log ""
Write-Log "  COMPILATION" "Cyan"
Write-SubSep

$allOk = $true

$allOk = $allOk -and (Invoke-Compile "activations"       "$NVCC $CFLAGS -c src/activations.cu -o $OBJDIR/activations.o")
$allOk = $allOk -and (Invoke-Compile "hadamard"          "$NVCC $CFLAGS -c src/hadamard.cu    -o $OBJDIR/hadamard.o")
$allOk = $allOk -and (Invoke-Compile "gemm"              "$NVCC $CFLAGS -c src/gemm.cu        -o $OBJDIR/gemm.o")
$allOk = $allOk -and (Invoke-Compile "conv1d"            "$NVCC $CFLAGS -c src/conv1d.cu      -o $OBJDIR/conv1d.o")
$allOk = $allOk -and (Invoke-Compile "scan1d"            "$NVCC $CFLAGS -c src/scan1d.cu      -o $OBJDIR/scan1d.o")
$allOk = $allOk -and (Invoke-Compile "scan2d/naive"      "$NVCC $CFLAGS -c src/scan2d/naive/scan2d_naive.cu             -o $OBJDIR/scan2d_naive.o")
$allOk = $allOk -and (Invoke-Compile "scan2d/naive_vec"  "$NVCC $CFLAGS -c src/scan2d/naive_vec/scan2d_naive_vec.cu     -o $OBJDIR/scan2d_naive_vec.o")
$allOk = $allOk -and (Invoke-Compile "scan2d/coop"       "$NVCC $CFLAGS -rdc=true -c src/scan2d/coop/scan2d_coop.cu    -o $OBJDIR/scan2d_coop.o")
$allOk = $allOk -and (Invoke-Compile "scan2d/tiled"      "$NVCC $CFLAGS -c src/scan2d/tiled/scan2d_tiled.cu            -o $OBJDIR/scan2d_tiled.o")

if (-not $allOk) {
    Write-Log ""
    Write-Log "  ARRET : erreurs de compilation. Voir rapport." "Red"
    $Log | Out-File -FilePath $ReportFile -Encoding utf8
    exit 1
}

# ── Linking des executables de test ───────────────────────────────────

Write-Log ""
Write-Log "  LINKING" "Cyan"
Write-SubSep

$p1_objs    = "$OBJDIR/activations.o $OBJDIR/hadamard.o"
$p3_objs    = "$OBJDIR/conv1d.o $OBJDIR/scan1d.o"
$scan2d_objs = "$OBJDIR/scan2d_naive.o $OBJDIR/scan2d_naive_vec.o $OBJDIR/scan2d_coop.o $OBJDIR/scan2d_tiled.o"

Invoke-Link "test_activations" "$NVCC $CFLAGS tests/test_activations.cu $OBJDIR/activations.o -o $OBJDIR/test_activations.exe" "$OBJDIR/test_activations.exe" | Out-Null
Invoke-Link "test_hadamard"    "$NVCC $CFLAGS tests/test_hadamard.cu    $OBJDIR/hadamard.o    -o $OBJDIR/test_hadamard.exe"    "$OBJDIR/test_hadamard.exe"    | Out-Null
Invoke-Link "test_gemm"        "$NVCC $CFLAGS tests/test_gemm.cu        $OBJDIR/gemm.o        -o $OBJDIR/test_gemm.exe"        "$OBJDIR/test_gemm.exe"        | Out-Null
Invoke-Link "test_conv1d"      "$NVCC $CFLAGS tests/test_conv1d.cu      $OBJDIR/conv1d.o      -o $OBJDIR/test_conv1d.exe"      "$OBJDIR/test_conv1d.exe"      | Out-Null
Invoke-Link "test_scan1d"      "$NVCC $CFLAGS tests/test_scan1d.cu      $OBJDIR/scan1d.o      -o $OBJDIR/test_scan1d.exe"      "$OBJDIR/test_scan1d.exe"      | Out-Null
Invoke-Link "test_scan2d"      "$NVCC $CFLAGS -rdc=true tests/test_scan2d.cu $scan2d_objs -lcudadevrt -o $OBJDIR/test_scan2d.exe" "$OBJDIR/test_scan2d.exe"  | Out-Null

if ($LinkErrors -gt 0) {
    Write-Log ""
    Write-Log "  ATTENTION : $LinkErrors executable(s) non produit(s)." "DarkYellow"
    Write-Log "  Les tests correspondants seront ignores." "DarkYellow"
}

# ── Execution des tests ───────────────────────────────────────────────

Write-Log ""
Write-Sep
Write-Log "  EXECUTION DES TESTS" "Magenta"
Write-Sep

Invoke-Test "Phase 1A — Activations"     "$OBJDIR/test_activations.exe"
Invoke-Test "Phase 1B — Hadamard"        "$OBJDIR/test_hadamard.exe"
Invoke-Test "Phase 2  — GEMV + GEMM"     "$OBJDIR/test_gemm.exe"
Invoke-Test "Phase 3A — Conv1D"          "$OBJDIR/test_conv1d.exe"
Invoke-Test "Phase 3B — Scan 1D"        "$OBJDIR/test_scan1d.exe"
Invoke-Test "Phase 4  — Scan 2D (x4)"   "$OBJDIR/test_scan2d.exe"

# ── Resume final ──────────────────────────────────────────────────────

Write-Log ""
Write-Sep
Write-Log "  RESUME FINAL" "Magenta"
Write-Sep

$success     = ($FailedTests -eq 0 -and $CompileErrors -eq 0 -and $LinkErrors -eq 0)
$statusMsg   = if ($success) { "SUCCES TOTAL" } else { "ECHECS DETECTES" }
$statusColor = if ($success) { "Green" } else { "Red" }

Write-Log ""
Write-Log "  Statut           : $statusMsg"                    $statusColor
Write-Log "  Tests passes     : $PassedTests / $TotalTests"    $(if ($PassedTests -eq $TotalTests) { "Green" } else { "Yellow" })
Write-Log "  Tests echoues    : $FailedTests"                  $(if ($FailedTests   -eq 0) { "Green" } else { "Red" })
Write-Log "  Erreurs compile  : $CompileErrors"                $(if ($CompileErrors -eq 0) { "Green" } else { "Red" })
Write-Log "  Erreurs link     : $LinkErrors"                   $(if ($LinkErrors    -eq 0) { "Green" } else { "Red" })
Write-Log ""
Write-Log "  Rapport sauve -> $ReportFile" "Cyan"
Write-Sep

# ── Sauvegarde ────────────────────────────────────────────────────────

$Log | Out-File -FilePath $ReportFile -Encoding utf8
