import streamlit as st
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import io
import os
import requests
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Chat History Viewer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION - Choose your connection method
# ============================================================================

# Option 1: API Endpoint (Recommended - Most Secure)
# Set this to your FastAPI server URL
API_BASE_URL = os.getenv("FASTAPI_URL", "http://20.55.97.82:8000")

# Option 2: Network Path (If VMs are on same network)
# Uncomment and set the network path to the database file
# NETWORK_DB_PATH = r"\\192.168.1.100\shared\chat_history.db"  # Windows
# NETWORK_DB_PATH = "/mnt/shared/chat_history.db"  # Linux

# Option 3: SSH Tunnel (Advanced - requires paramiko)
# SSH_HOST = os.getenv("SSH_HOST", "your-vm-ip")
# SSH_USER = os.getenv("SSH_USER", "username")
# SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "~/.ssh/id_rsa")
# REMOTE_DB_PATH = "/path/to/chat_history.db"

# Connection method selection
CONNECTION_METHOD = os.getenv("CONNECTION_METHOD", "api")  # Options: "api", "network", "ssh", "local"

# ============================================================================
# CONNECTION METHODS
# ============================================================================

def load_via_api(user_id_filter=None, start_date=None, end_date=None):
    """Load data via FastAPI endpoint (Recommended method)."""
    try:
        # Build query parameters
        params = {}
        if user_id_filter:
            params["user_id"] = user_id_filter
        if start_date:
            params["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y-%m-%d")
        
        # Build endpoint URL
        # If API_BASE_URL already includes the path, use it directly
        # Otherwise append /chat-history
        if API_BASE_URL.endswith('/chat-history'):
            endpoint_url = API_BASE_URL
        else:
            endpoint_url = f"{API_BASE_URL.rstrip('/')}/chat-history"
        
        # Call FastAPI endpoint
        response = requests.get(endpoint_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp_display'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Connection Error: {e}")
        st.info(f"Make sure FastAPI is running at: {API_BASE_URL}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data via API: {e}")
        return pd.DataFrame()

def load_via_network_path(user_id_filter=None, start_date=None, end_date=None):
    """Load data via network file path (Windows/Linux network share)."""
    import sqlite3
    
    try:
        # Use the network path
        db_path = os.getenv("NETWORK_DB_PATH", NETWORK_DB_PATH if 'NETWORK_DB_PATH' in globals() else None)
        
        if not db_path:
            st.error("❌ NETWORK_DB_PATH not configured")
            return pd.DataFrame()
        
        if not Path(db_path).exists():
            st.error(f"❌ Database not found at network path: {db_path}")
            st.info("💡 Make sure the network path is accessible and the database file exists")
            return pd.DataFrame()
        
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        
        query = "SELECT id, user_id, question, answer, intent, summary, timestamp FROM chat_history WHERE 1=1"
        params = []
        
        if user_id_filter:
            query += " AND user_id = ?"
            params.append(user_id_filter)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.strftime("%Y-%m-%d %H:%M:%S"))
        
        if end_date:
            end_datetime = datetime.combine(end_date, datetime.max.time())
            query += " AND timestamp <= ?"
            params.append(end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp_display'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df
    except Exception as e:
        st.error(f"❌ Network Path Error: {e}")
        return pd.DataFrame()

def load_via_ssh(user_id_filter=None, start_date=None, end_date=None):
    """Load data via SSH tunnel (Advanced - requires paramiko)."""
    try:
        import paramiko
        import sqlite3
        import tempfile
        
        ssh_host = os.getenv("SSH_HOST", SSH_HOST if 'SSH_HOST' in globals() else None)
        ssh_user = os.getenv("SSH_USER", SSH_USER if 'SSH_USER' in globals() else None)
        ssh_key_path = os.getenv("SSH_KEY_PATH", SSH_KEY_PATH if 'SSH_KEY_PATH' in globals() else None)
        remote_db_path = os.getenv("REMOTE_DB_PATH", REMOTE_DB_PATH if 'REMOTE_DB_PATH' in globals() else None)
        
        if not all([ssh_host, ssh_user, remote_db_path]):
            st.error("❌ SSH configuration incomplete")
            return pd.DataFrame()
        
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect via SSH key or password
        if ssh_key_path and Path(ssh_key_path).exists():
            ssh.connect(ssh_host, username=ssh_user, key_filename=ssh_key_path)
        else:
            # Fallback to password (not recommended for production)
            ssh_password = os.getenv("SSH_PASSWORD")
            if not ssh_password:
                st.error("❌ SSH key or password required")
                return pd.DataFrame()
            ssh.connect(ssh_host, username=ssh_user, password=ssh_password)
        
        # Copy database file via SFTP
        sftp = ssh.open_sftp()
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db_path = temp_db.name
        temp_db.close()
        
        sftp.get(remote_db_path, temp_db_path)
        sftp.close()
        ssh.close()
        
        # Read from local temp file
        conn = sqlite3.connect(temp_db_path, check_same_thread=False)
        
        query = "SELECT id, user_id, question, answer, intent, summary, timestamp FROM chat_history WHERE 1=1"
        params = []
        
        if user_id_filter:
            query += " AND user_id = ?"
            params.append(user_id_filter)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.strftime("%Y-%m-%d %H:%M:%S"))
        
        if end_date:
            end_datetime = datetime.combine(end_date, datetime.max.time())
            query += " AND timestamp <= ?"
            params.append(end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Clean up temp file
        os.unlink(temp_db_path)
        
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp_display'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df
    except ImportError:
        st.error("❌ paramiko not installed. Install with: pip install paramiko")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ SSH Connection Error: {e}")
        return pd.DataFrame()

def load_via_local(user_id_filter=None, start_date=None, end_date=None):
    """Load data from local database file."""
    import sqlite3
    
    db_path = "chat_history.db"
    if not Path(db_path).exists():
        db_path = Path(__file__).parent / "chat_history.db"
        if not db_path.exists():
            db_path = "./chat_history.db"
    
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        query = "SELECT id, user_id, question, answer, intent, summary, timestamp FROM chat_history WHERE 1=1"
        params = []
        
        if user_id_filter:
            query += " AND user_id = ?"
            params.append(user_id_filter)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.strftime("%Y-%m-%d %H:%M:%S"))
        
        if end_date:
            end_datetime = datetime.combine(end_date, datetime.max.time())
            query += " AND timestamp <= ?"
            params.append(end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp_display'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df
    except Exception as e:
        st.error(f"❌ Local Database Error: {e}")
        return pd.DataFrame()

# ============================================================================
# MAIN LOAD FUNCTION
# ============================================================================

@st.cache_data(ttl=5)
def load_chat_history(user_id_filter=None, start_date=None, end_date=None):
    """Main function to load chat history based on connection method."""
    if CONNECTION_METHOD == "api":
        return load_via_api(user_id_filter, start_date, end_date)
    elif CONNECTION_METHOD == "network":
        return load_via_network_path(user_id_filter, start_date, end_date)
    elif CONNECTION_METHOD == "ssh":
        return load_via_ssh(user_id_filter, start_date, end_date)
    else:
        return load_via_local(user_id_filter, start_date, end_date)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_unique_user_ids():
    """Get list of unique user IDs."""
    df = load_chat_history()
    if not df.empty and 'user_id' in df.columns:
        return sorted(df['user_id'].dropna().unique().tolist())
    return []

def get_date_range():
    """Get min and max dates from database."""
    df = load_chat_history()
    if not df.empty and 'timestamp' in df.columns:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        return min_date, max_date
    return None, None

# ============================================================================
# MAIN APP
# ============================================================================

st.title("💬 Chat History Viewer & Exporter")
st.markdown("---")

# Connection status
st.sidebar.header("🔌 Connection")
st.sidebar.info(f"**Method:** {CONNECTION_METHOD.upper()}")
if CONNECTION_METHOD == "api":
    endpoint_url = API_BASE_URL if API_BASE_URL.endswith('/chat-history') else f"{API_BASE_URL.rstrip('/')}/chat-history"
    st.sidebar.info(f"**API URL:** {endpoint_url}")
    st.sidebar.caption(f"Base URL: {API_BASE_URL}")

# Manual refresh button
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# User ID filter
user_ids = get_unique_user_ids()
if user_ids:
    selected_user = st.sidebar.selectbox(
        "Select User ID",
        options=["All"] + user_ids,
        index=0
    )
    user_id_filter = None if selected_user == "All" else selected_user
else:
    st.sidebar.info("No user IDs found")
    user_id_filter = None

# Date range filters
min_date, max_date = get_date_range()
if min_date and max_date:
    st.sidebar.markdown("### Date Range")
    
    start_date_filter = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date_filter = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    if start_date_filter > end_date_filter:
        st.sidebar.warning("⚠️ Start date must be before end date")
        start_date_filter = min_date
        end_date_filter = max_date
else:
    start_date_filter = None
    end_date_filter = None

# Load data
df = load_chat_history(
    user_id_filter=user_id_filter,
    start_date=start_date_filter,
    end_date=end_date_filter
)

# Display statistics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df))
if not df.empty:
    col2.metric("Unique Users", df['user_id'].nunique() if 'user_id' in df.columns else 0)
    col3.metric("Date Range", f"{df['timestamp'].min().date()}" if 'timestamp' in df.columns else "N/A")
    col4.metric("Latest Record", df['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S") if 'timestamp' in df.columns else "N/A")
else:
    col2.metric("Unique Users", 0)
    col3.metric("Date Range", "N/A")
    col4.metric("Latest Record", "N/A")

st.markdown("---")

# Display data
if df.empty:
    st.info("📭 No records found matching the selected filters.")
    if CONNECTION_METHOD == "api":
        st.warning("💡 Check if FastAPI server is running and accessible")
else:
    display_columns = ['id', 'user_id', 'timestamp_display', 'intent', 'question', 'answer']
    if 'summary' in df.columns:
        display_columns.append('summary')
    
    display_df = df[display_columns].copy() if all(col in df.columns for col in display_columns) else df
    
    display_df.columns = display_df.columns.str.replace('_', ' ').str.title()
    display_df.columns = display_df.columns.str.replace('Timestamp Display', 'Timestamp')
    
    st.subheader("📊 Chat History Table")
    
    search_term = st.text_input("🔍 Search in questions and answers", "")
    if search_term:
        mask = (
            display_df['Question'].str.contains(search_term, case=False, na=False) |
            display_df['Answer'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = display_df[mask]
        st.info(f"Found {len(filtered_df)} records matching '{search_term}'")
    else:
        filtered_df = display_df
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Export section
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Export to CSV", type="primary"):
            export_df = df.copy()
            if 'timestamp_display' in export_df.columns:
                export_df = export_df.drop(columns=['timestamp_display'])
            
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_string,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv"
            )
    
    with col2:
        st.info(f"📊 Export will contain {len(df)} records with {len(df.columns)} columns")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; padding: 10px;'>
        <small>💬 Chat History Viewer | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
    </div>
    """,
    unsafe_allow_html=True
)

