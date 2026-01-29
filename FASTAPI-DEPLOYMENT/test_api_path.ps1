# Test script to find correct API path
# Run: .\test_api_path.ps1

$baseUrl = "http://20.55.97.82:8000"
$paths = @(
    "/chat-history",
    "/project/chat-history",
    "/api/chat-history",
    "/deploy/chat-history",
    "/v1/chat-history",
    "/fastapi/chat-history"
)

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Testing different API paths for: $baseUrl" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

$foundPath = $null

foreach ($path in $paths) {
    $fullUrl = "$baseUrl$path"
    Write-Host "Testing: $fullUrl" -ForegroundColor White
    
    try {
        $response = Invoke-WebRequest -Uri $fullUrl -Method GET -TimeoutSec 5 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ SUCCESS! Status Code: 200" -ForegroundColor Green
            Write-Host "✅ Correct path is: $path" -ForegroundColor Green
            Write-Host ""
            Write-Host "Response preview (first 200 chars):" -ForegroundColor Yellow
            $preview = $response.Content.Substring(0, [Math]::Min(200, $response.Content.Length))
            Write-Host $preview -ForegroundColor Gray
            Write-Host ""
            Write-Host "=" * 60 -ForegroundColor Green
            Write-Host "UPDATE STREAMLIT APP:" -ForegroundColor Green
            Write-Host "Set API_BASE_URL = `"$baseUrl$(if ($path -ne '/chat-history') { $path.Replace('/chat-history', '') } else { '' })`"" -ForegroundColor Yellow
            Write-Host "=" * 60 -ForegroundColor Green
            $foundPath = $path
            break
        }
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode) {
            Write-Host "❌ Failed - Status Code: $statusCode" -ForegroundColor Red
        } else {
            Write-Host "❌ Failed - Connection error or timeout" -ForegroundColor Red
        }
    }
    Write-Host ""
}

if (-not $foundPath) {
    Write-Host "❌ None of the tested paths worked!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check if FastAPI server is running" -ForegroundColor White
    Write-Host "2. Check if port 8000 is accessible" -ForegroundColor White
    Write-Host "3. Try accessing: $baseUrl/docs in browser" -ForegroundColor White
    Write-Host "4. Check FastAPI logs for actual endpoint path" -ForegroundColor White
}



