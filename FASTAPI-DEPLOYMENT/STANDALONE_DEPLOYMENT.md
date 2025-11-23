# Standalone Streamlit Deployment Guide

## Overview

`streamlit_standalone.py` is a **single-file deployment** that includes both:
- ✅ Frontend UI (Streamlit)
- ✅ Backend API logic (from `rhl_fastapi_deploy.py`)

**No separate FastAPI server needed!** Everything runs in one Streamlit process.

---

## Quick Start

### Prerequisites

1. **Required files in same directory:**
   - `streamlit_standalone.py`
   - `rhl_fastapi_deploy.py`
   - `FILES/local_vectorstore/` (ChromaDB vector store)
   - `FILES/cache_questions.xlsx`
   - `FILES/video_link_topic.xlsx`

2. **Environment variables** (`.env` file or system):
   ```
   OPENAI_API_KEY=your_key
   GOOGLE_API_KEY=your_key
   ```

3. **Dependencies:**
   ```bash
   pip install streamlit fastapi uvicorn
   # ... and all other dependencies from requirements.txt
   ```

### Run Locally

```bash
cd FASTAPI-DEPLOYMENT
streamlit run streamlit_standalone.py
```

Open browser to: `http://localhost:8501`

---

## How It Works

1. **Imports backend functions** from `rhl_fastapi_deploy.py`
2. **Initializes all models** on first run (cached for performance)
3. **Calls backend functions directly** (no HTTP requests!)
4. **Displays results** in Streamlit UI

**Advantages:**
- ✅ Faster (no network overhead)
- ✅ Simpler deployment (one file/command)
- ✅ Easier debugging (single process)
- ✅ Lower resource usage (no separate API server)

---

## Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. **Push code to GitHub**
   - Include: `streamlit_standalone.py`, `rhl_fastapi_deploy.py`, `FILES/` folder

2. **Deploy on Streamlit Cloud:**
   - Go to: https://share.streamlit.io
   - Connect GitHub repo
   - Main file: `FASTAPI-DEPLOYMENT/streamlit_standalone.py`
   - Add secrets: `OPENAI_API_KEY`, `GOOGLE_API_KEY`

3. **Done!** Your app is live.

**Note:** Make sure `FILES/` folder is included in repo (or uploaded separately).

---

### Option 2: Self-Hosted (VPS/Server)

1. **Upload files to server**

2. **Install dependencies:**
   ```bash
   pip install streamlit -r requirements.txt
   ```

3. **Run with systemd:**
   Create `/etc/systemd/system/streamlit-standalone.service`:
   ```ini
   [Unit]
   Description=Streamlit Standalone Medical Chatbot
   After=network.target

   [Service]
   Type=simple
   User=your_user
   WorkingDirectory=/path/to/FASTAPI-DEPLOYMENT
   Environment="OPENAI_API_KEY=your_key"
   Environment="GOOGLE_API_KEY=your_key"
   ExecStart=/usr/local/bin/streamlit run streamlit_standalone.py --server.port 8501 --server.address 0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

4. **Start service:**
   ```bash
   sudo systemctl enable streamlit-standalone
   sudo systemctl start streamlit-standalone
   ```

---

### Option 3: Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy files
COPY FASTAPI-DEPLOYMENT/ .
COPY FILES/ ./FILES/
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit

# Expose port
EXPOSE 8501

# Run
CMD ["streamlit", "run", "streamlit_standalone.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t medical-chatbot-standalone .
docker run -p 8501:8501 -e OPENAI_API_KEY=xxx -e GOOGLE_API_KEY=xxx medical-chatbot-standalone
```

---

## Comparison: Standalone vs Separate Services

| Feature | Standalone | Separate (FastAPI + Streamlit) |
|---------|------------|-------------------------------|
| **Deployment** | One command | Two services |
| **Performance** | Faster (direct calls) | Slower (HTTP overhead) |
| **Complexity** | Simple | More complex |
| **Resource Usage** | Lower | Higher (2 processes) |
| **Scaling** | Limited | Better (can scale API separately) |
| **Best For** | Small-medium apps | Large-scale production |

---

## Troubleshooting

### "Error importing backend functions"
- ✅ Make sure `rhl_fastapi_deploy.py` is in same directory
- ✅ Check all imports are available

### "ChromaDB vector store not found"
- ✅ Run `setup_local_vectorstore.py` first
- ✅ Check `FILES/local_vectorstore/` exists

### "Models not initialized"
- ✅ Wait for initialization (30-60 seconds first time)
- ✅ Check environment variables are set
- ✅ Check internet connection (downloading models)

### Slow first response
- ✅ First run initializes models (normal)
- ✅ Subsequent requests are faster (models cached)

---

## Summary

**Deploy with one command:**
```bash
streamlit run streamlit_standalone.py
```

**That's it!** No separate FastAPI server, no complex setup, just one file to deploy! 🚀

