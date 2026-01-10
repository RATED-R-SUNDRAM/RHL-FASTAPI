import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import io

# Page configuration
st.set_page_config(
    page_title="Chat History Viewer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database path - check VM path first, then local paths
VM_DB_PATH = "/home/wowadmin/rhlai/RHL-FASTAPI/chat_history.db"
VM_DB_PATH_ALT = "/home/wowadmin/rhlai/RHL-FASTAPI/FASTAPI-DEPLOYMENT/chat_history.db"

# Try VM paths first (for VM deployment)
DB_PATH = None
if Path(VM_DB_PATH).exists():
    DB_PATH = str(VM_DB_PATH)
elif Path(VM_DB_PATH_ALT).exists():
    DB_PATH = str(VM_DB_PATH_ALT)
else:
    # Try alternative local paths if VM paths not found
    local_paths = [
        "chat_history.db",
        str(Path(__file__).parent / "chat_history.db"),
        str(Path(__file__).parent.parent / "chat_history.db"),
        "./chat_history.db"
    ]
    for path in local_paths:
        if Path(path).exists():
            DB_PATH = path
            break
    
    # If still not found, use VM path as default (will show error if doesn't exist)
    if DB_PATH is None:
        DB_PATH = VM_DB_PATH

@st.cache_data(ttl=5)  # Cache for 5 seconds for dynamic updates
def load_chat_history(user_id_filter=None, start_date=None, end_date=None):
    """
    Load chat history from database with optional filters.
    
    Args:
        user_id_filter: Filter by user_id (None for all)
        start_date: Filter by start date (datetime, None for no filter)
        end_date: Filter by end date (datetime, None for no filter)
    
    Returns:
        DataFrame with chat history
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        
        # Build query with filters
        query = "SELECT id, user_id, question, answer, intent, summary, timestamp FROM chat_history WHERE 1=1"
        params = []
        
        if user_id_filter:
            query += " AND user_id = ?"
            params.append(user_id_filter)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.strftime("%Y-%m-%d %H:%M:%S"))
        
        if end_date:
            # Add 23:59:59 to include the full end date
            end_datetime = datetime.combine(end_date, datetime.max.time())
            query += " AND timestamp <= ?"
            params.append(end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            # Convert timestamp to datetime for better display
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Format timestamp for display
            df['timestamp_display'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df
    
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_unique_user_ids():
    """Get list of unique user IDs from database."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT user_id FROM chat_history ORDER BY user_id")
        user_ids = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return user_ids
    except Exception as e:
        st.error(f"Error fetching user IDs: {e}")
        return []

def get_date_range():
    """Get min and max dates from database."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM chat_history")
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] and result[1]:
            min_date = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S").date()
            max_date = datetime.strptime(result[1], "%Y-%m-%d %H:%M:%S").date()
            return min_date, max_date
        return None, None
    except Exception as e:
        return None, None

# Main app
st.title("💬 Chat History Viewer & Exporter")
st.markdown("---")

# Check if database exists
if not Path(DB_PATH).exists():
    st.error(f"❌ Database not found at: {DB_PATH}")
    st.info("Please ensure the database file exists in the same directory or update DB_PATH in the script.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Manual refresh button
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Note about auto-refresh
st.sidebar.info("💡 Tip: Click 'Refresh Data' to see latest updates from database")

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
    st.sidebar.info("No user IDs found in database")
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
    
    # Validate date range
    if start_date_filter > end_date_filter:
        st.sidebar.warning("⚠️ Start date must be before end date")
        start_date_filter = min_date
        end_date_filter = max_date
else:
    start_date_filter = None
    end_date_filter = None

# Load data with filters
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
else:
    # Prepare display dataframe (without raw timestamp)
    display_columns = ['id', 'user_id', 'timestamp_display', 'intent', 'question', 'answer']
    if 'summary' in df.columns:
        display_columns.append('summary')
    
    display_df = df[display_columns].copy() if all(col in df.columns for col in display_columns) else df
    
    # Rename columns for better display
    display_df.columns = display_df.columns.str.replace('_', ' ').str.title()
    display_df.columns = display_df.columns.str.replace('Timestamp Display', 'Timestamp')
    
    st.subheader("📊 Chat History Table")
    
    # Search functionality
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
    
    # Display table with pagination
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
        # CSV download
        if st.button("📄 Export to CSV", type="primary"):
            # Prepare export dataframe (include all columns)
            export_df = df.copy()
            
            # Remove display timestamp column if exists
            if 'timestamp_display' in export_df.columns:
                export_df = export_df.drop(columns=['timestamp_display'])
            
            # Convert DataFrame to CSV
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Create download button
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_string,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv"
            )
    
    with col2:
        # Show export preview
        st.info(f"📊 Export will contain {len(df)} records with {len(df.columns)} columns")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <small>💬 Chat History Viewer | Last updated: {}</small>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)

