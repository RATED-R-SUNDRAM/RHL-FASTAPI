"""
FastAPI endpoint to serve chat history data for Streamlit app.
Add this endpoint to your existing FastAPI app.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import aiosqlite
import json

# Add this endpoint to your existing FastAPI app
# Example: Add to rhl_fastapi_deployment_clone.py

@app.get("/chat-history")
async def get_chat_history(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(1000, description="Maximum number of records to return")
) -> List[Dict[str, Any]]:
    """
    Get chat history with optional filters.
    This endpoint is used by the Streamlit viewer app.
    """
    try:
        # Build query
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
        
        # Execute query
        async with aiosqlite.connect("chat_history.db") as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = ['id', 'user_id', 'question', 'answer', 'intent', 'summary', 'timestamp']
            results = []
            for row in rows:
                result = {}
                for i, col in enumerate(columns):
                    result[col] = row[i]
                results.append(result)
            
            return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Optional: Add CORS if Streamlit is on different domain
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, specify your Streamlit domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


