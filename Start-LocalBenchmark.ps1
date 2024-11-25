Write-Host "[Compile]"
cargo build --release
Move-Item ../target/release/ahc039 . -Force
Write-Host "[Run]"
$env:DURATION_MUL = "1.5"
dotnet marathon run-local
#./relative_score.exe -d ./data/results -o min