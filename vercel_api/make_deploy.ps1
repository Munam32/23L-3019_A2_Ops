param([string]$Out = "D:\vercel_deploy")

$root = Split-Path -Parent $PSScriptRoot
if (Test-Path $Out) {
    Get-ChildItem $Out -Force | Where-Object { $_.Name -ne ".vercel" } | Remove-Item -Recurse -Force
}
New-Item -ItemType Directory -Force "$Out\api\model" | Out-Null
Copy-Item "$PSScriptRoot\index.py", "$PSScriptRoot\onnx_infer.py", "$PSScriptRoot\restore_path.py" "$Out\api"
Copy-Item "$root\api.py" "$Out\api\service.py"
Copy-Item "$root\validation.py" "$Out\api"
Copy-Item "$root\models\champion.onnx", "$root\models\champion_onnx.json" "$Out\api\model"
Copy-Item "$PSScriptRoot\vercel.json", "$PSScriptRoot\requirements.txt" $Out
Get-ChildItem $Out -Recurse -File | Select-Object @{n="File";e={$_.FullName.Substring($Out.Length)}}, @{n="KB";e={[math]::Round($_.Length/1KB)}}
