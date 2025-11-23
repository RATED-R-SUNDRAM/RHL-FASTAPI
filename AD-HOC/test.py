from dotenv import load_dotenv
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI

import pandas as pd 

# Load the video data
df = pd.read_excel("D:\\RHL-WH\\RHL-FASTAPI\\FILES\\video_link.xlsx")
print("Original data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Extract video topics from description
def extract_video_topic(description):
    """Extract video topics from description"""
    if pd.isna(description):
        return ""
    
    # Find keywords section
    keywords_start = description.find("keywords")
    if keywords_start != -1:
        return description[keywords_start + 10:].strip()
    return ""

# Apply extraction
df['video_topic'] = df['Description'].apply(extract_video_topic)

# Clean up the data
df = df[df['video_topic'] != ""]  # Remove rows without topics
df = df.dropna(subset=['video_topic'])  # Remove NaN values

print(f"\nAfter processing: {df.shape[0]} videos with topics")
print("\nSample video topics:")
print(df[['video_topic']].head(10))

# Save processed data
df.to_excel("D:\\RHL-WH\\RHL-FASTAPI\\FILES\\video_link_topic.xlsx", index=False)
print(f"\nSaved processed data to video_link_topic.xlsx")

#%% 
