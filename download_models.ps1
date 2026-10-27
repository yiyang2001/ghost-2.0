# Re-download every model file that is not stored in this repository (about 3.7 GB).
# Run from anywhere:   powershell -ExecutionPolicy Bypass -File .\download_models.ps1
# Files that already exist with the right size are skipped, so it is safe to re-run or resume.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$release = 'https://github.com/ai-forever/ghost-2.0/releases/download/aligner'

# relative path, source url, expected size in bytes
$models = @(
	@('aligner_checkpoints/aligner_1020_gaze_final.ckpt', "$release/aligner_1020_gaze_final.ckpt", 1476894254),
	@('blender_checkpoints/blender_lama.ckpt', "$release/blender_lama.ckpt", 1056939521),
	@('weights/backbone50_1.pth', "$release/backbone50_1.pth", 174680546),
	@('weights/segformer_B5_ce.onnx', "$release/segformer_B5_ce.onnx", 329893835),
	@('weights/big-lama.pt', 'https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt', 205803670),
	@('weights/mask2former-swin-tiny-coco-instance/config.json', 'https://huggingface.co/facebook/mask2former-swin-tiny-coco-instance/resolve/main/config.json', 79763),
	@('.torch/hub/checkpoints/vgg19-d01eb7cb.pth', "$release/vgg19-d01eb7cb.pth", 574689280),
	@('repos/stylematte/stylematte/checkpoints/stylematte_synth.pth', 'https://github.com/chroneus/stylematte/releases/download/weights/stylematte_synth.pth', 140040541)
)

# stylematte code is needed next to its checkpoint (repos/ is not tracked in this repository)
$stylematte = Join-Path $root 'repos/stylematte'
if (-not (Test-Path (Join-Path $stylematte 'stylematte/models.py'))) {
	if (Test-Path $stylematte) { Remove-Item $stylematte -Recurse -Force }
	New-Item -ItemType Directory -Force (Split-Path $stylematte) | Out-Null
	git clone https://github.com/chroneus/stylematte.git $stylematte
	if ($LASTEXITCODE -ne 0) { throw 'git clone stylematte failed' }
	git -C $stylematte checkout 77af912d6619e26132c613b81f50b3766dd44142
}

$failed = @()
foreach ($model in $models) {
	$path = Join-Path $root $model[0]
	if ((Test-Path $path) -and (Get-Item $path).Length -eq $model[2]) {
		Write-Host "[skip] $($model[0])"
		continue
	}
	New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
	Write-Host "[download] $($model[0])"
	$part = "$path.part"
	curl.exe -L --fail --retry 5 --retry-delay 3 -C - -o $part $model[1]
	if ($LASTEXITCODE -eq 0 -and (Get-Item $part).Length -eq $model[2]) {
		Move-Item $part $path -Force
	} else {
		Write-Host "[failed] $($model[0]) (re-run the script to resume)" -ForegroundColor Red
		$failed += $model[0]
	}
}

if ($failed.Count -gt 0) { Write-Host "$($failed.Count) file(s) failed." -ForegroundColor Red; exit 1 }
Write-Host 'All models are in place.' -ForegroundColor Green
