# Connecting Streamlit to Database on Separate VM

This guide explains how to connect your Streamlit app to a database running on a separate VM.

## 🎯 Quick Solution Summary

**Recommended: API Endpoint Method** (Easiest & Most Secure)
- Add an endpoint to your FastAPI app
- Streamlit calls the API to get data
- No direct database access needed

---

## 📋 Method 1: API Endpoint (RECOMMENDED)

### ✅ Advantages
- ✅ Most secure (no direct DB access)
- ✅ Works across any network
- ✅ Easy to implement
- ✅ Can add authentication easily

### Setup Steps

#### Step 1: Add Endpoint to FastAPI

Add this to your `rhl_fastapi_deployment_clone.py` file (after line 3128):

```python
@app.get("/chat-history")
async def get_chat_history(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(1000, description="Maximum number of records")
) -> List[Dict[str, Any]]:
    """Get chat history with optional filters."""
    try:
        query = "SELECT id, user_id, question, answer, intent, summary, timestamp FROM chat_history WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(f"{start_date} 00:00:00")
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(f"{end_date} 23:59:59")
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect("chat_history.db") as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            
            columns = ['id', 'user_id', 'question', 'answer', 'intent', 'summary', 'timestamp']
            results = []
            for row in rows:
                result = {col: row[i] for i, col in enumerate(columns)}
                results.append(result)
            
            return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
```

**Add imports at the top if not present:**
```python
from typing import Optional, List, Dict, Any
```

#### Step 2: Enable CORS (if Streamlit is on different domain)

Add this after creating your FastAPI app:

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

#### Step 3: Configure Streamlit App

1. Use the file: `chat_history_viewer_remote.py`

2. Set environment variable:
```bash
export FASTAPI_URL="http://YOUR_VM_IP:8000"
export CONNECTION_METHOD="api"
```

Or create a `.env` file:
```
FASTAPI_URL=http://192.168.1.100:8000
CONNECTION_METHOD=api
```

3. Run Streamlit:
```bash
streamlit run chat_history_viewer_remote.py
```

---

## 📋 Method 2: Network File Path (Same Network)

### ✅ Advantages
- ✅ Direct database access
- ✅ Fast (no API overhead)
- ✅ Simple if VMs are on same network

### ⚠️ Requirements
- Both VMs must be on same network
- Network file sharing enabled
- Database file must be accessible via network path

### Setup Steps

#### For Windows VMs:

1. **On FastAPI VM:**
   - Share the folder containing `chat_history.db`
   - Note the network path: `\\VM_IP\shared\chat_history.db`

2. **On Streamlit VM:**
   - Map network drive or use UNC path
   - Set environment variable:
   ```bash
   export NETWORK_DB_PATH="\\192.168.1.100\shared\chat_history.db"
   export CONNECTION_METHOD="network"
   ```

#### For Linux VMs:

1. **On FastAPI VM:**
   - Install Samba or NFS
   - Share the directory containing database
   - Mount point: `/mnt/shared/chat_history.db`

2. **On Streamlit VM:**
   - Mount the shared directory
   - Set environment variable:
   ```bash
   export NETWORK_DB_PATH="/mnt/shared/chat_history.db"
   export CONNECTION_METHOD="network"
   ```

---

## 📋 Method 3: SSH Tunnel (Advanced)

### ✅ Advantages
- ✅ Secure (encrypted connection)
- ✅ Works over internet
- ✅ No need to expose database

### ⚠️ Requirements
- SSH access to FastAPI VM
- SSH key or password
- `paramiko` library installed

### Setup Steps

1. **Install paramiko:**
```bash
pip install paramiko
```

2. **Set environment variables:**
```bash
export SSH_HOST="your-vm-ip"
export SSH_USER="username"
export SSH_KEY_PATH="~/.ssh/id_rsa"  # or use SSH_PASSWORD
export REMOTE_DB_PATH="/path/to/chat_history.db"
export CONNECTION_METHOD="ssh"
```

3. **Run Streamlit:**
```bash
streamlit run chat_history_viewer_remote.py
```

---

## 🔧 Configuration Summary

### Environment Variables

| Variable | Method | Description |
|----------|--------|-------------|
| `CONNECTION_METHOD` | All | `"api"`, `"network"`, `"ssh"`, or `"local"` |
| `FASTAPI_URL` | API | FastAPI server URL (e.g., `http://192.168.1.100:8000`) |
| `NETWORK_DB_PATH` | Network | Network path to database file |
| `SSH_HOST` | SSH | SSH hostname/IP |
| `SSH_USER` | SSH | SSH username |
| `SSH_KEY_PATH` | SSH | Path to SSH private key |
| `SSH_PASSWORD` | SSH | SSH password (if not using key) |
| `REMOTE_DB_PATH` | SSH | Path to database on remote VM |

### Quick Setup Script

Create `setup_streamlit.sh`:

```bash
#!/bin/bash
# Set your FastAPI VM IP
export FASTAPI_URL="http://YOUR_VM_IP:8000"
export CONNECTION_METHOD="api"

# Run Streamlit
streamlit run chat_history_viewer_remote.py
```

---

## 🧪 Testing Connection

### Test API Method:
```bash
curl http://YOUR_VM_IP:8000/chat-history
```

### Test Network Path:
```bash
# Windows
dir \\VM_IP\shared\chat_history.db

# Linux
ls /mnt/shared/chat_history.db
```

### Test SSH:
```bash
ssh username@VM_IP "ls /path/to/chat_history.db"
```

---

## 🔒 Security Recommendations

1. **API Method:**
   - Add authentication token
   - Use HTTPS in production
   - Restrict CORS origins

2. **Network Method:**
   - Use VPN for network access
   - Restrict file permissions
   - Use read-only access

3. **SSH Method:**
   - Use SSH keys (not passwords)
   - Disable password authentication
   - Use non-standard SSH port

---

## 🐛 Troubleshooting

### API Method Issues:

**Problem:** Connection refused
- Check FastAPI is running
- Verify firewall allows port 8000
- Check VM IP address

**Problem:** CORS errors
- Add CORS middleware to FastAPI
- Check allowed origins

### Network Method Issues:

**Problem:** Cannot access network path
- Verify network connectivity
- Check file sharing permissions
- Test with `ping` and `telnet`

### SSH Method Issues:

**Problem:** Authentication failed
- Verify SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
- Test SSH connection manually
- Check SSH service is running

---

## 📝 Quick Start (Recommended)

1. **Add endpoint to FastAPI** (copy from `chat_history_api_endpoint.py`)
2. **Set environment variable:**
   ```bash
   export FASTAPI_URL="http://YOUR_VM_IP:8000"
   export CONNECTION_METHOD="api"
   ```
3. **Run Streamlit:**
   ```bash
   streamlit run chat_history_viewer_remote.py
   ```

Done! ✅


