# How to Find and Configure the Correct API Path

## 🔍 Step 1: Find the Correct Path

Your FastAPI might be deployed with a path prefix like:
- `/project/chat-history`
- `/api/chat-history`
- `/deploy/chat-history`
- Or any other prefix

### Method 1: Check FastAPI Server Logs

When your FastAPI starts, check the logs. It might show:
```
Application startup complete.
Uvicorn running on http://0.0.0.0:8000/prefix
```

### Method 2: Test Different Paths with curl

Try these common paths:

```bash
# Test 1: No prefix (default)
curl http://20.55.97.82:8000/chat-history

# Test 2: /project prefix
curl http://20.55.97.82:8000/project/chat-history

# Test 3: /api prefix
curl http://20.55.97.82:8000/api/chat-history

# Test 4: /deploy prefix
curl http://20.55.97.82:8000/deploy/chat-history

# Test 5: Check root
curl http://20.55.97.82:8000/
curl http://20.55.97.82:8000/docs
```

### Method 3: Test from Browser

Open these URLs in browser:
- http://20.55.97.82:8000/docs (FastAPI docs - shows all endpoints)
- http://20.55.97.82:8000/chat-history
- http://20.55.97.82:8000/project/chat-history
- http://20.55.97.82:8000/api/chat-history

---

## ✅ What to Look For

### Success Response (200 OK):
```json
[
  {
    "id": 1,
    "user_id": "user123",
    "question": "...",
    "answer": "...",
    ...
  }
]
```

### Error Responses:
- `404 Not Found` - Path doesn't exist, try different prefix
- `Connection refused` - Server not running or wrong IP/port
- `Timeout` - Firewall blocking or server down

---

## 🔧 Step 2: Update Streamlit App with Correct Path

Once you find the correct path, update the Streamlit app:

### Option 1: Update Code Directly

Open `chat_history_viewer_remote.py` and find line ~24:

```python
API_BASE_URL = os.getenv("FASTAPI_URL", "http://20.55.97.82:8000")
```

Change to include the path:
```python
# If path is /project/chat-history
API_BASE_URL = os.getenv("FASTAPI_URL", "http://20.55.97.82:8000/project")

# OR if path is /api/chat-history
API_BASE_URL = os.getenv("FASTAPI_URL", "http://20.55.97.82:8000/api")
```

Then the code will call: `http://20.55.97.82:8000/project/chat-history`

### Option 2: Use Environment Variable (Recommended)

Set environment variable with full base URL (without `/chat-history`):

```bash
# Windows PowerShell
$env:FASTAPI_URL="http://20.55.97.82:8000/project"

# Linux/Mac
export FASTAPI_URL="http://20.55.97.82:8000/project"
```

The code will append `/chat-history` automatically.

---

## 🧪 Step 3: Test Commands

### Windows PowerShell Commands:

```powershell
# Test basic path
Invoke-WebRequest -Uri "http://20.55.97.82:8000/chat-history" -Method GET

# Test with /project prefix
Invoke-WebRequest -Uri "http://20.55.97.82:8000/project/chat-history" -Method GET

# Test with verbose output (shows status code)
Invoke-WebRequest -Uri "http://20.55.97.82:8000/project/chat-history" -Method GET | Select-Object StatusCode, Content

# Get only content
(Invoke-WebRequest -Uri "http://20.55.97.82:8000/project/chat-history").Content

# Check if endpoint exists (fast test)
Test-NetConnection -ComputerName 20.55.97.82 -Port 8000
```

### Linux/Mac Commands:

```bash
# Test basic path
curl http://20.55.97.82:8000/chat-history

# Test with /project prefix
curl http://20.55.97.82:8000/project/chat-history

# Test with verbose output
curl -v http://20.55.97.82:8000/project/chat-history

# Test and show only status code
curl -s -o /dev/null -w "%{http_code}" http://20.55.97.82:8000/project/chat-history

# Test and show full response
curl -s http://20.55.97.82:8000/project/chat-history | head -20
```

---

## 📝 Example: If Path is `/project/chat-history`

### Step 1: Verify Path Works
```bash
# Test from terminal
curl http://20.55.97.82:8000/project/chat-history

# Should return JSON array with chat history
```

### Step 2: Update Streamlit Configuration

Edit `chat_history_viewer_remote.py`, line ~24:
```python
# Before:
API_BASE_URL = os.getenv("FASTAPI_URL", "http://20.55.97.82:8000")

# After:
API_BASE_URL = os.getenv("FASTAPI_URL", "http://20.55.97.82:8000/project")
```

### Step 3: Verify Code Calls Correct Path

The code already appends `/chat-history` to the base URL:
```python
response = requests.get(f"{API_BASE_URL}/chat-history", ...)
```

So with `API_BASE_URL = "http://20.55.97.82:8000/project"`, it will call:
- `http://20.55.97.82:8000/project/chat-history` ✅

---

## 🔍 How to Check Your FastAPI Deployment Path

### Check if FastAPI uses mount/prefix:

In your FastAPI code, look for:
```python
# Common patterns:
app = FastAPI()  # No prefix
subapi = FastAPI()
app.mount("/project", subapi)  # Has /project prefix
app.mount("/api", subapi)  # Has /api prefix

# Or in deployment:
# uvicorn app:app --root-path /project
```

### Check Deployment Configuration:

Look for:
- `--root-path` flag in uvicorn command
- Reverse proxy configuration (nginx, apache)
- Docker/deployment config files
- Environment variables like `ROOT_PATH` or `API_PREFIX`

---

## 🎯 Quick Test Script

Create `test_api_path.ps1` (Windows PowerShell):

```powershell
$baseUrl = "http://20.55.97.82:8000"
$paths = @(
    "/chat-history",
    "/project/chat-history",
    "/api/chat-history",
    "/deploy/chat-history",
    "/v1/chat-history"
)

Write-Host "Testing different API paths..." -ForegroundColor Yellow

foreach ($path in $paths) {
    $fullUrl = "$baseUrl$path"
    Write-Host "`nTesting: $fullUrl" -ForegroundColor Cyan
    
    try {
        $response = Invoke-WebRequest -Uri $fullUrl -Method GET -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ SUCCESS! Path is: $path" -ForegroundColor Green
            Write-Host "Response preview:" -ForegroundColor Green
            $response.Content.Substring(0, [Math]::Min(200, $response.Content.Length))
            break
        }
    } catch {
        Write-Host "❌ Failed (Status: $($_.Exception.Response.StatusCode.value__))" -ForegroundColor Red
    }
}
```

Run it:
```powershell
.\test_api_path.ps1
```

---

## 📋 Summary

1. **Find correct path** using curl or browser
2. **Update `API_BASE_URL`** in `chat_history_viewer_remote.py` to include prefix
3. **Test again** with curl to verify
4. **Run Streamlit** - it should work now

The key is: If your API is at `/project/chat-history`, set:
```python
API_BASE_URL = "http://20.55.97.82:8000/project"
```
Not:
```python
API_BASE_URL = "http://20.55.97.82:8000/project/chat-history"
```

Because the code adds `/chat-history` automatically.



