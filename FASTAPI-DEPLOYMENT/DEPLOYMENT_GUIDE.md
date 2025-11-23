# Complete Deployment Guide: FastAPI Backend + Streamlit Frontend

## Architecture Overview
- **FastAPI Backend** (`rhl_fastapi_deploy.py`): Provides `/chat` and `/chat-stream` API endpoints
- **Streamlit Frontend** (`streamlit_stream_demo.py`): Makes HTTP requests to FastAPI backend

Both components MUST be deployed separately and accessible via URLs.

---

## Option 1: Railway.app (Recommended - Easy)

### Part A: Deploy FastAPI Backend

1. **Install Railway CLI:**
   ```bash
   npm i -g @railway/cli
   ```

2. **Login to Railway:**
   ```bash
   railway login
   ```

3. **Initialize Railway project:**
   ```bash
   cd FASTAPI-DEPLOYMENT
   railway init
   ```

4. **Set environment variables in Railway dashboard:**
   - `OPENAI_API_KEY` = your OpenAI key
   - `GOOGLE_API_KEY` = your Google API key

5. **Deploy:**
   ```bash
   railway up
   ```

6. **Get your FastAPI URL:**
   - Railway will provide a URL like: `https://your-app-name.railway.app`
   - **Save this URL!** You'll need it for Streamlit

---

### Part B: Deploy Streamlit Frontend

**Option B1: Streamlit Cloud (Free)**

1. **Push your code to GitHub** (if not already)

2. **Go to:** https://share.streamlit.io

3. **Click "New app"**

4. **Configure:**
   - Repository: Your GitHub repo
   - Branch: `main`
   - Main file path: `FASTAPI-DEPLOYMENT/streamlit_stream_demo.py`

5. **Advanced Settings → Secrets:**
   ```
   FASTAPI_URL=https://your-app-name.railway.app
   ```
   (Use the FastAPI URL from Part A)

6. **Click "Deploy!"**

7. **Share the Streamlit URL** with users!

---

**Option B2: Railway for Streamlit (Paid)**

1. **Create new service in Railway**

2. **Set environment variable:**
   ```
   FASTAPI_URL=https://your-fastapi-url.railway.app
   ```

3. **Start command:**
   ```
   streamlit run FASTAPI-DEPLOYMENT/streamlit_stream_demo.py --server.port $PORT --server.address 0.0.0.0
   ```

---

## Option 2: Render.com (Free Tier)

### Part A: Deploy FastAPI Backend

1. **Create `render.yaml` in project root:**
   ```yaml
   services:
     - type: web
       name: medical-chatbot-api
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: cd FASTAPI-DEPLOYMENT && uvicorn rhl_fastapi_deploy:app --host 0.0.0.0 --port $PORT
       envVars:
         - key: OPENAI_API_KEY
           sync: false
         - key: GOOGLE_API_KEY
           sync: false
   ```

2. **Push to GitHub**

3. **Go to render.com → New Web Service**

4. **Connect repo → Render auto-detects `render.yaml`**

5. **Get FastAPI URL:** `https://medical-chatbot-api.onrender.com`

---

### Part B: Deploy Streamlit on Streamlit Cloud

1. **Deploy on Streamlit Cloud** (same as Option 1 Part B)

2. **Set secret:**
   ```
   FASTAPI_URL=https://medical-chatbot-api.onrender.com
   ```

---

## Option 3: Quick Testing with ngrok

### Part A: Start FastAPI Locally

1. **Terminal 1 - Start FastAPI:**
   ```bash
   cd FASTAPI-DEPLOYMENT
   python rhl_fastapi_deploy.py
   ```

2. **Terminal 2 - Expose with ngrok:**
   ```bash
   ngrok http 8000
   ```
   - Copy the HTTPS URL: `https://abc123.ngrok.io`
   - **This is your FastAPI URL!**

---

### Part B: Deploy Streamlit

**Option 1: Streamlit Cloud**
1. Deploy on Streamlit Cloud
2. Set secret: `FASTAPI_URL=https://abc123.ngrok.io`

**Option 2: Local + ngrok**
1. **Terminal 3 - Start Streamlit:**
   ```bash
   cd FASTAPI-DEPLOYMENT
   streamlit run streamlit_stream_demo.py
   ```

2. **Terminal 4 - Expose Streamlit:**
   ```bash
   ngrok http 8501
   ```
   - Copy URL: `https://xyz789.ngrok.io`
   - **Share this with users!**
   - Users set API URL in sidebar: `https://abc123.ngrok.io`

---

## Option 4: Local Network (Same WiFi)

### Part A: Start FastAPI

```bash
cd FASTAPI-DEPLOYMENT
python rhl_fastapi_deploy.py
```

- FastAPI runs on: `http://YOUR_IP:8000`

---

### Part B: Start Streamlit

```bash
cd FASTAPI-DEPLOYMENT
streamlit run streamlit_stream_demo.py --server.address 0.0.0.0
```

- Streamlit runs on: `http://YOUR_IP:8501`
- Share this URL with others on same WiFi
- They set API URL in sidebar: `http://YOUR_IP:8000`

---

## Quick Reference

### FastAPI Deployment Commands

**Railway:**
```bash
railway login
railway init
railway up
```

**Render:**
- Push to GitHub, connect to Render, auto-deploy

**Local:**
```bash
python rhl_fastapi_deploy.py
```

---

### Streamlit Deployment Commands

**Streamlit Cloud:**
- Push to GitHub → share.streamlit.io → Deploy

**Local:**
```bash
streamlit run streamlit_stream_demo.py
```

**Local (accessible on network):**
```bash
streamlit run streamlit_stream_demo.py --server.address 0.0.0.0
```

---

## Important Notes

1. **FastAPI must be deployed FIRST** - Streamlit needs its URL
2. **Set `FASTAPI_URL` environment variable** in Streamlit deployment
3. **Or users can manually enter** FastAPI URL in Streamlit sidebar
4. **Both services must be accessible** from the internet (or same network)

---

## Troubleshooting

### "Connection refused" error in Streamlit
- ✅ Check FastAPI is running
- ✅ Verify FastAPI URL is correct
- ✅ Test FastAPI endpoint directly: `https://your-api-url.com/test-debug`

### Streamlit can't find FastAPI
- ✅ Check `FASTAPI_URL` environment variable is set
- ✅ Or enter FastAPI URL manually in Streamlit sidebar
- ✅ Ensure FastAPI URL is accessible (no firewall blocking)

---

## Recommended Setup

**For Quick Testing:**
- FastAPI: Local + ngrok
- Streamlit: Streamlit Cloud (set ngrok URL as secret)

**For Production:**
- FastAPI: Railway.app or Render.com
- Streamlit: Streamlit Cloud (set production API URL as secret)

