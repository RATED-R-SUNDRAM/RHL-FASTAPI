# Setup Instructions for Chat History Viewer

## Your VM Information
- **FastAPI VM IP:** `20.55.97.82`
- **FastAPI Port:** `8000` (default, change if different)
- **Connection Method:** API Endpoint (Recommended)

---

## ✅ Checklist: What You Need

### 1. On FastAPI VM (20.55.97.82)

#### Required:
- [ ] FastAPI server is running
- [ ] Port 8000 is accessible (firewall allows it)
- [ ] `/chat-history` endpoint is added (already done in your code)
- [ ] Database file `chat_history.db` exists and is accessible
- [ ] FastAPI server restarted after adding endpoint

#### Test API Endpoint:
```bash
curl http://20.55.97.82:8000/chat-history
```

**Expected Response:** JSON array with chat history records

---

### 2. On Streamlit VM

#### Required Files:
- [ ] `chat_history_viewer_remote.py` (copy from this repo)
- [ ] `streamlit` installed: `pip install streamlit pandas requests`

#### Required Configuration:
- [ ] Environment variable set: `FASTAPI_URL=http://20.55.97.82:8000`
- [ ] Environment variable set: `CONNECTION_METHOD=api`

#### Run Streamlit:
```bash
streamlit run chat_history_viewer_remote.py
```

---

## 🔧 Configuration Steps

### Step 1: On FastAPI VM (20.55.97.82)

1. **Check if endpoint exists:**
   - Open `rhl_fastapi_deployment_clone.py`
   - Verify `/chat-history` endpoint is present (around line 3136)
   - If not, add it from the code provided

2. **Restart FastAPI server:**
   ```bash
   # Stop current server (Ctrl+C)
   # Restart with:
   uvicorn rhl_fastapi_deployment_clone:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Test endpoint:**
   ```bash
   curl http://20.55.97.82:8000/chat-history
   # OR from browser: http://20.55.97.82:8000/chat-history
   ```

4. **Check firewall (if connection fails):**
   ```bash
   # Allow port 8000
   sudo ufw allow 8000
   # OR
   sudo firewall-cmd --permanent --add-port=8000/tcp
   sudo firewall-cmd --reload
   ```

---

### Step 2: On Streamlit VM

1. **Copy the remote viewer file:**
   ```bash
   # Make sure chat_history_viewer_remote.py is on Streamlit VM
   ```

2. **Set environment variables:**

   **Option A: Set in terminal (temporary):**
   ```bash
   export FASTAPI_URL="http://20.55.97.82:8000"
   export CONNECTION_METHOD="api"
   ```

   **Option B: Create .env file (permanent):**
   ```bash
   # Create .env file
   echo "FASTAPI_URL=http://20.55.97.82:8000" > .env
   echo "CONNECTION_METHOD=api" >> .env
   ```

   **Option C: Edit in code (simple):**
   - Open `chat_history_viewer_remote.py`
   - Find line ~17: `API_BASE_URL = os.getenv("FASTAPI_URL", "http://your-fastapi-vm-ip:8000")`
   - Change to: `API_BASE_URL = os.getenv("FASTAPI_URL", "http://20.55.97.82:8000")`

3. **Install dependencies:**
   ```bash
   pip install streamlit pandas requests
   ```

4. **Run Streamlit:**
   ```bash
   streamlit run chat_history_viewer_remote.py
   ```

5. **Access in browser:**
   - Usually at: `http://localhost:8501`
   - Or check the URL shown in terminal

---

## 🧪 Testing Connection

### Test 1: API Endpoint (from any machine)
```bash
curl http://20.55.97.82:8000/chat-history
```

**Success:** Should return JSON array with chat records
**Failure:** Check firewall, FastAPI server status

### Test 2: From Streamlit VM
```bash
curl http://20.55.97.82:8000/chat-history
```

**Success:** Should return JSON array
**Failure:** Check network connectivity between VMs

### Test 3: Streamlit App
- Open Streamlit app in browser
- Check sidebar shows connection method: "API"
- Check sidebar shows API URL: "http://20.55.97.82:8000"
- Data should load automatically

---

## 🔒 Security Checklist

### Production Recommendations:

1. **Add Authentication:**
   - Add API key to `/chat-history` endpoint
   - Require token in request headers

2. **Use HTTPS:**
   - Set up SSL certificate
   - Change URL to: `https://20.55.97.82:8000`

3. **Restrict CORS:**
   - In FastAPI, specify your Streamlit domain
   - Don't use `allow_origins=["*"]` in production

4. **Firewall Rules:**
   - Only allow port 8000 from Streamlit VM IP
   - Don't expose port 8000 to public internet

---

## 🐛 Troubleshooting

### Problem: "Connection refused" or "Cannot connect"

**Solution:**
1. Check FastAPI is running: `ps aux | grep uvicorn`
2. Check port is listening: `netstat -tuln | grep 8000`
3. Check firewall: `sudo ufw status`
4. Test locally on FastAPI VM: `curl http://localhost:8000/chat-history`

### Problem: "CORS error" in browser

**Solution:**
Add CORS middleware to FastAPI:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Problem: "No data displayed" in Streamlit

**Solution:**
1. Check API response: `curl http://20.55.97.82:8000/chat-history`
2. Check Streamlit logs for errors
3. Verify database has records
4. Check filters aren't too restrictive

### Problem: "Database not found"

**Solution:**
1. Verify `chat_history.db` exists on FastAPI VM
2. Check file permissions
3. Check database path in FastAPI code

---

## 📋 Quick Setup Script (Streamlit VM)

Create `setup_streamlit.sh`:

```bash
#!/bin/bash
echo "Setting up Streamlit Chat History Viewer..."

# Set environment variables
export FASTAPI_URL="http://20.55.97.82:8000"
export CONNECTION_METHOD="api"

# Install dependencies
pip install streamlit pandas requests

# Test API connection
echo "Testing API connection..."
curl http://20.55.97.82:8000/chat-history

# Run Streamlit
echo "Starting Streamlit..."
streamlit run chat_history_viewer_remote.py
```

Make executable and run:
```bash
chmod +x setup_streamlit.sh
./setup_streamlit.sh
```

---

## ✅ Final Checklist

### FastAPI VM (20.55.97.82):
- [ ] FastAPI server running
- [ ] `/chat-history` endpoint accessible
- [ ] Port 8000 open in firewall
- [ ] Database file accessible
- [ ] Test: `curl http://20.55.97.82:8000/chat-history` works

### Streamlit VM:
- [ ] `chat_history_viewer_remote.py` copied
- [ ] Environment variable `FASTAPI_URL=http://20.55.97.82:8000` set
- [ ] Environment variable `CONNECTION_METHOD=api` set
- [ ] Dependencies installed (`streamlit`, `pandas`, `requests`)
- [ ] Can connect to API: `curl http://20.55.97.82:8000/chat-history`
- [ ] Streamlit app runs successfully

---

## 🎯 One-Line Test Commands

### From FastAPI VM:
```bash
curl http://localhost:8000/chat-history | head -20
```

### From Streamlit VM:
```bash
curl http://20.55.97.82:8000/chat-history | head -20
```

### From Any Machine (if port is open):
```bash
curl http://20.55.97.82:8000/chat-history | head -20
```

---

## 📞 Need Help?

If connection fails:
1. Check FastAPI server logs
2. Check firewall rules
3. Verify IP address is correct
4. Test with curl from Streamlit VM
5. Check network connectivity: `ping 20.55.97.82`


