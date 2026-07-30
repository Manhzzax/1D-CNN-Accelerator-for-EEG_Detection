param(
    [string]$OutputPath = "$PSScriptRoot\..\docs\assets\chbmit_detection_benchmark.png"
)

Add-Type -AssemblyName System.Drawing

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$width = 2400
$height = 1390
$bitmap = [System.Drawing.Bitmap]::new($width, $height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([System.Drawing.Color]::White)

$navy = [System.Drawing.Color]::FromArgb(15, 34, 58)
$teal = [System.Drawing.Color]::FromArgb(0, 112, 122)
$lightTeal = [System.Drawing.Color]::FromArgb(226, 244, 243)
$lightBlue = [System.Drawing.Color]::FromArgb(234, 241, 248)
$lightGray = [System.Drawing.Color]::FromArgb(243, 245, 247)
$dark = [System.Drawing.Color]::FromArgb(28, 36, 45)
$muted = [System.Drawing.Color]::FromArgb(83, 96, 110)
$border = [System.Drawing.Color]::FromArgb(189, 198, 207)

$titleFont = [System.Drawing.Font]::new("Arial", 33, [System.Drawing.FontStyle]::Bold)
$subtitleFont = [System.Drawing.Font]::new("Arial", 15, [System.Drawing.FontStyle]::Regular)
$groupFont = [System.Drawing.Font]::new("Arial", 15, [System.Drawing.FontStyle]::Bold)
$headerFont = [System.Drawing.Font]::new("Arial", 14, [System.Drawing.FontStyle]::Bold)
$cellFont = [System.Drawing.Font]::new("Arial", 13, [System.Drawing.FontStyle]::Regular)
$cellBoldFont = [System.Drawing.Font]::new("Arial", 13, [System.Drawing.FontStyle]::Bold)
$footerFont = [System.Drawing.Font]::new("Arial", 12, [System.Drawing.FontStyle]::Regular)

function Draw-TextCell {
    param(
        [string]$Text,
        [System.Drawing.Font]$Font,
        [System.Drawing.Color]$Color,
        [float]$X,
        [float]$Y,
        [float]$Width,
        [float]$Height,
        [System.Drawing.StringAlignment]$Alignment = [System.Drawing.StringAlignment]::Near
    )

    $format = [System.Drawing.StringFormat]::new()
    $format.Alignment = $Alignment
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $format.Trimming = [System.Drawing.StringTrimming]::EllipsisWord
    $format.FormatFlags = [System.Drawing.StringFormatFlags]::LineLimit
    $rect = [System.Drawing.RectangleF]::new($X + 10, $Y + 4, $Width - 20, $Height - 8)
    $graphics.DrawString($Text, $Font, [System.Drawing.SolidBrush]::new($Color), $rect, $format)
    $format.Dispose()
}

function Draw-Row {
    param(
        [object[]]$Values,
        [float]$Y,
        [float]$Height,
        [System.Drawing.Color]$Fill,
        [bool]$Bold = $false
    )

    $x = 70
    for ($i = 0; $i -lt $script:columnWidths.Count; $i++) {
        $cellWidth = $script:columnWidths[$i]
        $graphics.FillRectangle([System.Drawing.SolidBrush]::new($Fill), $x, $Y, $cellWidth, $Height)
        $graphics.DrawRectangle([System.Drawing.Pen]::new($script:border, 1), $x, $Y, $cellWidth, $Height)
        $alignment = if ($i -in @(2, 3, 4, 5, 6)) { [System.Drawing.StringAlignment]::Center } else { [System.Drawing.StringAlignment]::Near }
        $font = if ($Bold) { $script:cellBoldFont } else { $script:cellFont }
        Draw-TextCell -Text $Values[$i] -Font $font -Color $script:dark -X $x -Y $Y -Width $cellWidth -Height $Height -Alignment $alignment
        $x += $cellWidth
    }
}

$columnWidths = @(265, 430, 115, 185, 185, 130, 170, 295, 360)
$headers = @("Study", "Protocol / representation", "Channels", "Window accuracy", "Event sensitivity", "FAR / h", "Detection delay", "Model / deployment evidence", "How to read it")
$lineBreak = [Environment]::NewLine

Draw-TextCell -Text "CHB-MIT SEIZURE DETECTION BENCHMARK" -Font $titleFont -Color $navy -X 70 -Y 28 -Width 2260 -Height 54
Draw-TextCell -Text "Detection only. Window accuracy, seizure-event sensitivity, FAR/h, and delay are separate metrics." -Font $subtitleFont -Color $muted -X 70 -Y 82 -Width 2260 -Height 32

$graphics.FillRectangle([System.Drawing.SolidBrush]::new($lightTeal), 70, 126, 2260, 42)
Draw-TextCell -Text "CURRENT HARDWARE REFERENCE: run_21_raw_2s_temporal3 | 17 channels | 2 s at 256 Hz | locked chronological validation | INT16 package verified" -Font $groupFont -Color $teal -X 82 -Y 128 -Width 2235 -Height 36

$y = 190
$x = 70
for ($i = 0; $i -lt $columnWidths.Count; $i++) {
    $cellWidth = $columnWidths[$i]
    $graphics.FillRectangle([System.Drawing.SolidBrush]::new($navy), $x, $y, $cellWidth, 68)
    $graphics.DrawRectangle([System.Drawing.Pen]::new($navy, 1), $x, $y, $cellWidth, 68)
    Draw-TextCell -Text $headers[$i] -Font $headerFont -Color ([System.Drawing.Color]::White) -X $x -Y $y -Width $cellWidth -Height 68 -Alignment ([System.Drawing.StringAlignment]::Center)
    $x += $cellWidth
}
$y += 68

$graphics.FillRectangle([System.Drawing.SolidBrush]::new($lightBlue), 70, $y, 2260, 34)
Draw-TextCell -Text "A. CONTINUOUS EVENT DETECTION - PRIMARY CLINICAL COMPARISON" -Font $groupFont -Color $navy -X 82 -Y $y -Width 2230 -Height 34
$y += 34

$eventRows = @(
    @("Current run_21", "Shared model; 2 s raw window; 1:1 validation windows; causal 10-of-20", "17", "90.07%", "79.31% (23/29)", "0.467", "17 s median", "5,013 params${lineBreak}10.0 KB INT16 tensors", "Current screening reference${lineBreak}Validation only; not final test"),
    @("Shoeb and Guttag, 2010", "Patient-specific continuous detection; 24 cases, 173 test seizures", "Full", "NR", "96.00%", "0.08", "50% <3 s${lineBreak}mean 4.6 s", "NR", "Historical event comparator"),
    @("Chung et al., 2024${lineBreak}public labels", "Patient-specific k-fold; 13 selected cases; 4 s single-channel CNN", "1", "94.93 +/- 8.35%", "97.69 +/- 6.96%", "0.16 +/- 0.26", "8.0 +/- 9.4 s", "Parameter count NR", "Primary low-channel comparator"),
    @("Chung et al., 2024${lineBreak}reviewed labels", "Patient-specific k-fold; clinician re-annotated labels; 4 s single-channel CNN", "1", "98.18 +/- 1.83%", "99.62 +/- 1.39%", "0.22 +/- 0.34", "3.3 +/- 5.5 s", "Parameter count NR", "Context only: labels differ")
)
foreach ($row in $eventRows) {
    $fill = if ($row[0] -eq "Current run_21") { $lightTeal } else { [System.Drawing.Color]::White }
    Draw-Row -Values $row -Y $y -Height 106 -Fill $fill -Bold ($row[0] -eq "Current run_21")
    $y += 106
}

$graphics.FillRectangle([System.Drawing.SolidBrush]::new($lightGray), 70, $y, 2260, 34)
Draw-TextCell -Text "B. WINDOW CLASSIFICATION AND DEPLOYMENT CONTEXT - NOT A SUBSTITUTE FOR EVENT METRICS" -Font $groupFont -Color $navy -X 82 -Y $y -Width 2230 -Height 34
$y += 34

$classificationRows = @(
    @("Kashefi Amiri et al., 2025", "24 subjects; stratified 10-fold CV; DWT + 1D CNN-LSTM", "NR", "96.94 +/- 1.22%", "NR", "NR", "NR", "0.35 M params${lineBreak}1.67-30.7 M FLOPs", "Classification-only comparator"),
    @("Cao et al., 2025", "23 cases; DWT feature fusion + SVM-RFE + CNN-Bi-LSTM", "NR", "98.43%", "NR", "NR", "NR", "Parameter count NR${lineBreak}Heavy recurrent model", "Classification-only comparator"),
    @("Ahlawat et al., 2026${lineBreak}preprint baseline", "18-channel 1D-CNN; described on 686 EDF; split needs audit", "18", "96.17%", "NR", "NR", "NR", "1.63 MB FP32${lineBreak}0.39 ms reported CPU", "Preprint; not event-level"),
    @("Ahlawat et al., 2026${lineBreak}preprint pruned", "8-channel 1D-CNN; 2:4 structured sparsity", "8", "95.15%", "NR", "NR", "NR", "0.44 MB INT8${lineBreak}50% sparse weights", "Preprint; efficiency context")
)
foreach ($row in $classificationRows) {
    Draw-Row -Values $row -Y $y -Height 106 -Fill ([System.Drawing.Color]::White)
    $y += 106
}

$graphics.FillRectangle([System.Drawing.SolidBrush]::new($navy), 70, $y + 24, 2260, 76)
Draw-TextCell -Text "READING RULE: Compare run_21 directly only with detection studies that report event sensitivity, FAR/h, and delay. Accuracy rows use different windows, labels, sampling ratios, channels, and splits; they show context, not a numerical ranking." -Font $footerFont -Color ([System.Drawing.Color]::White) -X 94 -Y ($y + 28) -Width 2210 -Height 68
Draw-TextCell -Text "Sources: P02 Chung 2024; P05 Kashefi 2025; P06 Cao 2025; P12 Shoeb 2010; P19 Ahlawat 2026. Full source/page map: docs/chbmit_literature_benchmark_tables.md" -Font $footerFont -Color $muted -X 70 -Y ($y + 112) -Width 2260 -Height 26 -Alignment ([System.Drawing.StringAlignment]::Center)

$bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()

Write-Output "Wrote $OutputPath"
