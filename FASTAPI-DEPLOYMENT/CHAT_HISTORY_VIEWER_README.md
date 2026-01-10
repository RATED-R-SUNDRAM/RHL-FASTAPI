# Chat History Viewer - Streamlit App

A simple Streamlit application to view and export chat history from the FastAPI database.

## Features

- 📊 **View Chat History**: Display all questions and answers stored in the database
- 🔍 **Advanced Filtering**: Filter by User ID, Start Date, and End Date
- 🔎 **Search Functionality**: Search through questions and answers
- 📥 **CSV Export**: Download filtered results as CSV file
- 🔄 **Auto-refresh**: Option to auto-refresh data every 5 seconds
- 📈 **Statistics**: Display total records, unique users, and date ranges

## Prerequisites

- Python 3.8+
- Streamlit
- Pandas
- SQLite3 (included with Python)

## Installation

1. Install required packages:
```bash
pip install streamlit pandas
```

Or use the existing requirements:
```bash
pip install -r requirements_streamlit.txt
```

## Usage

### Local Deployment

1. Make sure `chat_history.db` exists in the same directory as the script (or update `DB_PATH` in the script)

2. Run the Streamlit app:
```bash
streamlit run FASTAPI-DEPLOYMENT/chat_history_viewer.py
```

3. Open your browser to `http://localhost:8501`

### Cloud Deployment (Streamlit Cloud)

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set the main file path: `FASTAPI-DEPLOYMENT/chat_history_viewer.py`
5. Deploy!

### Cloud Deployment (Other Platforms)

#### Railway / Heroku / Render

1. Create a `Procfile`:
```
web: streamlit run chat_history_viewer.py --server.port=$PORT --server.address=0.0.0.0
```

2. Ensure the database file is accessible (you may need to use environment variables for the path)

3. Deploy with your platform's instructions

## Configuration

### Database Path

If your database is in a different location, update the `DB_PATH` variable in the script:

```python
DB_PATH = "chat_history.db"  # Relative path
# OR
DB_PATH = "/path/to/your/chat_history.db"  # Absolute path
```

### Auto-refresh Interval

To change the auto-refresh interval, modify the `ttl` parameter in the `@st.cache_data` decorator:

```python
@st.cache_data(ttl=5)  # Change 5 to desired seconds
```

## Features Explained

### Filters

- **User ID**: Select a specific user or "All" to see all users
- **Start Date**: Filter records from this date onwards
- **End Date**: Filter records up to this date
- **Search**: Search for keywords in questions and answers

### Export

- Click "Export to CSV" button
- Click "Download CSV" to download the filtered results
- CSV includes all columns: id, user_id, question, answer, intent, summary, timestamp

### Auto-refresh

- Enable "Auto-refresh every 5 seconds" checkbox in sidebar
- Data will automatically refresh every 5 seconds
- Use "Refresh Data" button for manual refresh

## Database Schema

The app expects a SQLite database with the following schema:

```sql
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    question TEXT,
    answer TEXT,
    intent TEXT,
    summary TEXT DEFAULT '',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

## Troubleshooting

### Database Not Found

- Check that `chat_history.db` exists in the correct location
- Update `DB_PATH` variable if database is elsewhere
- For cloud deployment, ensure database file is included in deployment

### No Data Displayed

- Verify database has records
- Check filters aren't too restrictive
- Ensure database file has correct permissions

### Export Not Working

- Make sure there are records in the filtered results
- Check browser allows downloads
- Verify CSV buffer size isn't too large (consider pagination for very large datasets)

## Notes

- The app uses caching for performance (5-second TTL)
- Large datasets may take time to load - consider adding pagination if needed
- CSV exports include all filtered records - for very large exports, consider splitting by date range


