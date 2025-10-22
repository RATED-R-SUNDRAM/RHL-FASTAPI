from dotenv import load_dotenv
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # This loads variables from .env file
GOOGLE_API_KEY = 'AIzaSyBFEkKZB6tm_ojt5H6XNobOyDugMlFm9bw'

# Debug: Check if the key is loaded
print(f"GOOGLE_API_KEY loaded: {GOOGLE_API_KEY is not None}")
print(f"GOOGLE_API_KEY value: {GOOGLE_API_KEY[:10] if GOOGLE_API_KEY else 'None'}...")

if GOOGLE_API_KEY:
    google_genai = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",api_key=GOOGLE_API_KEY)
    print("ChatGoogleGenerativeAI initialized successfully!")
    
    # Ask a question using the model
    print("\n" + "="*50)
    print("ASKING A QUESTION TO GEMINI")
    print("="*50)
    
    question = "What are the symptoms of jaundice in newborns? answer in around 100-150 words"
    print(f"Question: {question}")
    print("\nAnswer:")
    
    try:
        a=time.time()
        response = google_genai.invoke(question)
        b=time.time()
        print(f"Time taken: {b-a} seconds")
        print(response.content)
    except Exception as e:
        print(f"Error asking question: {e}")
        
else:
    print("ERROR: GOOGLE_API_KEY not found in environment variables!")
    print("Make sure you have GOOGLE_API_KEY=your_key_here in your .env file")