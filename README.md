# 🩺 RHL-API: Intelligent Medical Chatbot with Hybrid RAG & FastAPI 🚀

Welcome to the RHL-API project – your intelligent companion for medical queries! This project powers a sophisticated Medical Chatbot designed to deliver accurate, context-aware answers to your health questions. Built with a robust Retrieval Augmented Generation (RAG) architecture and served via FastAPI, it's ready to elevate your medical information experience.

## ✨ Project Highlights & Objective

Our mission is simple: **to provide precise and personalized medical information.** This chatbot doesn't just pull facts; it understands your intent, refines your questions, intelligently searches vast medical knowledge, and synthesizes answers tailored to your needs. Plus, it remembers your conversations with a per-user chat history, making every interaction smarter.

## 🧠 The Brains Behind the Bot: Hybrid RAG Architecture

The chatbot operates on a meticulously designed multi-stage RAG pipeline, ensuring both speed and accuracy. Here’s a peek under the hood:

1.  **🔍 Intent Classification: What Are You Asking?**
    *   Every user query kicks off with a quick assessment! An advanced LLM (powered by `gpt-4o-mini`) intelligently categorizes your input as a `MEDICAL_QUESTION`, a `FOLLOW_UP` to a previous answer, or simple `CHITCHAT`.
    *   This initial step is crucial for directing your query to the right processing path.

2.  **✨ Query Refinement: Making Your Questions Crystal Clear**
    *   For all medical and follow-up questions, another LLM (`gpt-4o-mini`) steps in to polish your query:
        *   **Contextual Rephrasing**: Turns vague follow-ups (e.g., "tell me more") into clear, standalone questions using your chat history.
        *   **Abbreviation Expansion**: Deciphers medical acronyms (e.g., "IUFD" becomes "Intrauterine fetal death") for better understanding.
        *   **Spelling Correction**: Fixes typos, ensuring every search hits the mark.
    *   This step ensures your question is perfectly optimized for retrieval!

3.  **📚 Hybrid Retrieval: Finding the Needle in the Haystack**
    *   We don't just search; we intelligently retrieve! The refined query dives into a Pinecone vector database, combining:
        *   **Vector Search**: For lightning-fast semantic similarity matching.
        *   **Cross-Encoder Re-ranking**: A specialized medical-domain cross-encoder model then meticulously re-scores the retrieved chunks, boosting the most relevant information to the top.

4.  **⚖️ Expert Judging: Is This Information Good Enough?**
    *   Each potential answer chunk undergoes a rigorous evaluation by a dedicated LLM (`gpt-4o-mini`). It asks: "Is this relevant? Is it sufficient to form a strong answer?"
    *   Our smart two-tier sorting system prioritizes chunks: first by `topic_match` strength (strong 💪 > medium > absolutely_not_possible 🚫), then by the cross-encoder score.
    *   The top 4 qualified chunks are chosen for crafting your answer, with the next 2 set aside for clever follow-up suggestions.

5.  **💬 Contextual Answer Synthesis: Crafting Your Perfect Response**
    *   The chosen chunks are then handed over to the main LLM (`gpt-4o-mini`), which synthesizes a concise, factual, and easy-to-understand answer.
    *   **Strict Rules Apply**: Answers are *only* based on the provided context, include source citations (e.g., "**According to [Source Document Name]**"), are presented in clear bullet points, kept under 150 words, and seamlessly integrate with your ongoing conversation.
    *   If a great follow-up question was identified, it's gracefully appended to your answer!

6.  **🔄 Smart Chat History: Your Personal Medical Journal**
    *   Every conversation is important! Your per-user chat history (question, answer, intent, and a summary) is securely stored in an SQLite database.
    *   To keep LLM context efficient, older chat turns are intelligently summarized by the `gpt-4o-mini` LLM. Crucially, `CHITCHAT` conversations are recorded for display but *never* used to influence the context of medical queries.

### 📊 Architectural Flow Diagram

```
+------------------+     +-------------------------------+
|    User Query    | --> |       FastAPI Backend         |
+------------------+     +-------------------------------+
         |                             |
         V                             V
+-------------------------------+   +---------------------------------+
| Retrieve Chat History (SQLite)|   | Construct LLM Chat Context      |
+-------------------------------+   +---------------------------------+
         |                                     |
         V                                     V
+-----------------------------------------------------------------+
|      1. Classify Message (LLM - gpt-4o-mini)                  |
|          (Outputs: MEDICAL_QUESTION, FOLLOW_UP, CHITCHAT)       |
+-----------------------------------------------------------------+
         |
         +----[ If CHITCHAT ]----------------------------------+
         |                                                    |
         V                                                    V
+-------------------------------+
| 💬 Handle Chitchat (LLM)       |      +---------------------------------+
+-------------------------------+
         |                                                    |
         V                                                    V
+-------------------------------+      +---------------------------------+
| 💾 Save History (SQLite)        |
+-------------------------------+
         |                                                    |
         +----------------------------------------------------+
                                      V
                          Return Chitchat Response
                                      
         +----[ If MEDICAL_QUESTION / FOLLOW_UP ]-------------+
         |                                                    |
         V                                                    V
+-----------------------------------------------------------------+
|      2. Reformulate Query (LLM - gpt-4o-mini)                 |
+-----------------------------------------------------------------+
         |
         V
+-----------------------------------------------------------------+
|      3. Hybrid Retrieval (Vector Search + Cross-Encoder Reranking)|
+-----------------------------------------------------------------+
         |
         V
+-----------------------------------------------------------------+
|      4. Document Judging (LLM - gpt-4o-mini)                  |
+-----------------------------------------------------------------+
         |
         V
+-----------------------------------------------------------------+
|      5. Synthesize Answer (LLM - gpt-4o-mini)                 |
+-----------------------------------------------------------------+
         |
         V
+-----------------------------------------------------------------+
|      💾 Save History (SQLite)                                  |
+-----------------------------------------------------------------+
         |
         V
+-----------------------------------------------------------------+
|      Return Medical Response + Follow-up Suggestion           |
+-----------------------------------------------------------------+
```

## 📂 Project Structure: Where Everything Lives

Here's how our codebase is neatly organized:

*   **`Models/`**: The heart of our RAG pipeline's intelligence.
    *   `final_model_local.py`: This is the Streamlit application code for **local testing and deployment** of the RAG pipeline.
    *   `new_architecture_*.py`: Various iterations and experimental versions of the RAG pipeline components.
*   **`deployment/`**: Where our API services reside.
    *   `rhl_fastapi_deploy.py`: The production-ready FastAPI backend that serves the medical chatbot.
    *   `main.py`: A primary entry point script, if needed for other deployment setups.
    *   `rhl_fastapi_v2.py`: Another version or iteration of the FastAPI code.
*   **`ad-hoc/`**: A home for standalone utilities, experimental scripts, or older Streamlit apps not part of the main FastAPI deployment.
    *   `abbr.py`, `ingest_lcc.py`, `rhl_fastapi.py`, `streamlit_app_local.py`, `streamlit_app.py`, `test.py`, `test2.py`
*   **`files/`**: All your valuable data resources.
    *   `abbr_dict.json`, `jan_july-26.json`, `railway.json`, `test.json`, `test_data.xlsx`, `~$test_data.xlsx`, `test2.csv` (and any other `.csv`, `.xlsx`, `.json` files)
*   **`chat_history.db`**: The SQLite database for persistent, per-user chat history.
*   **`requirements.txt`**: All Python dependencies required for the project.
*   **`.env`**: Your confidential environment variables (API keys, etc. - **remember to keep this file secure and out of version control!**).

## 🚀 Get Started: Running the Chatbot

Ready to fire up the chatbot? Follow these simple steps!

### 1. 🛠️ Setup: Prepare Your Environment

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd RHL-API
    ```
2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Secure Your Secrets (.env file):**
    Create a `.env` file in the root of your project with your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    PINECONE_API_KEY_NEW="your_pinecone_api_key_here"
    PINECONE_INDEX="medical-chatbot-index" # Or your specific Pinecone index name
    ```

### 2. ⚡ Running the FastAPI Backend (The API Service)

This is the core API service that powers the chatbot.

*   **Location:** `deployment/rhl_fastapi_deploy:app`
*   **Command:**
    ```bash
    uvicorn FASTAPI-DEPLOYMENT.rhl_fastapi_deploy:app --reload --host 0.0.0.0 --port 8000
    ```
*   Once running, the API will be accessible at `http://localhost:8000`.

### 3. 🧪 Testing the FastAPI Endpoint

After starting the FastAPI server, you can test its functionality:

*   **Interactive API Documentation (Swagger UI):**
    *   Simply open your web browser and navigate to: `http://localhost:8000/docs`
    *   You can directly interact with the `/chat` endpoint from this user-friendly interface!

*   **Direct URL (GET Requests):**
    *   You can also send `GET` requests directly via your browser or tools like `curl`.
    *   **Example for a Medical Question:**
        ```
        http://localhost:8000/chat?user_id=YOUR_UNIQUE_ID&message=What%20are%20the%20symptoms%20of%20jaundice?
        ```
        (Remember to replace `YOUR_UNIQUE_ID` and ensure your message is URL-encoded for spaces/special characters, though browsers usually handle this.)
    *   **Example for Chitchat:**
        ```
        http://localhost:8000/chat?user_id=YOUR_UNIQUE_ID&message=hi
        ```

### 4. 🌐 Accessing the Streamlit Cloud Deployment

Your Streamlit application, which demonstrates the RAG architecture, is deployed live!

*   **Application Code:** `Models/final_model_local.py`
*   To run it locally use the command <streamlit run Models/final_model_local.py> in Terminal.
*   **Live URL:** [https://rhl-reranker-architecture.streamlit.app/](https://rhl-reranker-architecture.streamlit.app/)
*   **Description:** This interactive UI allows you to test the core RAG pipeline locally and experience its capabilities in a user-friendly format.
