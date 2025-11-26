from flask import Flask, request, session, jsonify, make_response
from langchain_core.messages import HumanMessage, AIMessage
from flask_cors import CORS
from dotenv import load_dotenv
from flask_session import Session
import os
import logging
import traceback
import base64
import zipfile
from huggingface_hub import hf_hub_download
from langchain_google_genai import ChatGoogleGenerativeAI

# --- IMPORTS FROM YOUR SRC FOLDER ---
# Ensure your src/ folder is inside the backend directory
from src.medical_swarm import run_medical_swarm
from src.utils import load_rag_system, standardize_query, get_standalone_question, parse_agent_response, markdown_bold_to_html

# --- CONFIGURATION ---
DATA_DIR = "/data" # Or "./data" for local dev if /data doesn't exist
DB_DIR = os.path.join(DATA_DIR, "chroma_db")
UPLOAD_DIR = os.path.join(DATA_DIR, "Uploads")
SESSION_DIR = os.path.join(DATA_DIR, "flask_session")

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# --- DATABASE SETUP ---
def setup_database():
    """Downloads DB if not exists."""
    DATASET_REPO_ID = "WanIrfan/atlast-db"
    ZIP_FILENAME = "chroma_db.zip"
    
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        logger.info(f"✅ Database exists at {DB_DIR}.")
        return

    logger.info("📥 Downloading database...")
    try:
        # Create local fallback if /data doesn't exist (e.g. local windows dev)
        if not os.path.exists(DATA_DIR) and DATA_DIR == "/data":
             # Fallback to local folder if root /data is not accessible
             os.makedirs("data", exist_ok=True)
        else:
             os.makedirs(DATA_DIR, exist_ok=True)

        zip_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=ZIP_FILENAME, repo_type="dataset")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        logger.info("✅ Database setup complete!")
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")

setup_database()

# --- APP SETUP ---
app = Flask(__name__)
app.secret_key = "super_secret_key" # Change this in prod

# ✅ CORS CONFIGURATION (CRITICAL FOR NEXT.JS)
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

# Session Config
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_FILE_DIR"] = SESSION_DIR
# Ensure session dir exists
os.makedirs(SESSION_DIR, exist_ok=True)
Session(app)

# --- LLM & RAG SETUP ---
google_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.05, google_api_key=google_api_key)

logger.info("🌟 Loading RAG Systems...")
try:
    rag_systems = {
        'medical': load_rag_system("medical_csv_Agentic_retrieval", "medical", persist_directory=DB_DIR),
        'islamic': load_rag_system("islamic_texts_Agentic_retrieval", "islamic", persist_directory=DB_DIR),
        'insurance': load_rag_system("etiqa_Agentic_retrieval", "insurance", persist_directory=DB_DIR)
    }
except Exception as e:
    logger.error(f"❌ RAG Load Failed: {e}")
    rag_systems = {'medical': None, 'islamic': None, 'insurance': None}

# --- HELPER FUNCTIONS ---
def hydrate_history(raw_list):
    """Dicts -> LangChain Messages"""
    history = []
    if not raw_list: return history
    for item in raw_list:
        if item.get('type') == 'human': history.append(HumanMessage(content=item.get('content', '')))
        elif item.get('type') == 'ai': history.append(AIMessage(content=item.get('content', '')))
    return history

def dehydrate_history(messages):
    """LangChain Messages -> Dicts"""
    return [{'type': 'human' if isinstance(m, HumanMessage) else 'ai', 'content': m.content} for m in messages]

# --- API ROUTES ---

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

# --- GENERIC HANDLER FOR ALL DOMAINS ---
# This reduces code duplication. You can call it from specific routes.
def handle_chat_request(domain, session_hist_key, session_resp_key, system_key):
    # GET: Return History
    if request.method == "GET":
        return jsonify({
            "history": session.get(session_hist_key, []),
            "latest_response": session.get(session_resp_key, {})
        })

    # POST: Process Chat
    if request.method == "POST":
        try:
            # 1. Parse Inputs
            query = standardize_query(request.form.get("query", ""))
            has_image = 'image' in request.files and request.files['image'].filename
            has_doc = 'document' in request.files and request.files['document'].filename
            
            # 2. Setup History
            raw_history = session.get(session_hist_key, [])
            history_objs = hydrate_history(raw_history)
            
            answer, thoughts, validation, source = "", "", "", ""
            
            # 3. Logic Routing
            if has_doc and domain == 'medical':
                 # ... (Your Swarm Logic) ...
                 file = request.files['document']
                 doc_text = file.read().decode("utf-8") # Simplified for brevity
                 session['current_medical_document'] = doc_text
                 swarm_ans = run_medical_swarm(doc_text, query)
                 answer = markdown_bold_to_html(swarm_ans)
                 thoughts = "Swarm Analysis Complete"
                 source = "Medical Swarm"
                 
                 history_objs.append(HumanMessage(content=f"[Document Uploaded] {query}"))
                 history_objs.append(AIMessage(content=answer))

            elif has_image:
                # ... (Your Vision Logic) ...
                file = request.files['image']
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                path = os.path.join(UPLOAD_DIR, file.filename)
                file.save(path)
                
                with open(path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                
                # Vision call...
                vision_prompt = f"Analyze image context for query: {query}"
                msg = HumanMessage(content=[{"type": "text", "text": vision_prompt}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}])
                vis_pred = llm.invoke([msg]).content
                
                enhanced_q = f"User Query: {query} Image Context: {vis_pred}"
                agent = rag_systems[system_key]
                resp = agent.answer(enhanced_q, chat_history=history_objs)
                answer, thoughts, validation, source = parse_agent_response(resp)
                
                history_objs.append(HumanMessage(content=query))
                history_objs.append(AIMessage(content=answer))
                
            elif query:
                # ... (Standard RAG) ...
                agent = rag_systems[system_key]
                
                # Medical doc context check
                if domain == 'medical' and session.get('current_medical_document'):
                    history_objs.insert(0, HumanMessage(content=f"Context Document: {session['current_medical_document']}"))
                
                std_query = get_standalone_question(query, history_objs, llm)
                resp = agent.answer(std_query, chat_history=history_objs)
                answer, thoughts, validation, source = parse_agent_response(resp)
                
                history_objs.append(HumanMessage(content=query))
                history_objs.append(AIMessage(content=answer))
            
            # 4. Save & Return
            new_hist_dict = dehydrate_history(history_objs)
            session[session_hist_key] = new_hist_dict
            
            latest_resp = {
                'answer': answer, 'thoughts': thoughts, 
                'validation': validation, 'source': source
            }
            session[session_resp_key] = latest_resp
            
            return jsonify({
                "status": "success",
                "history": new_hist_dict,
                "latest_response": latest_resp
            })

        except Exception as e:
            logger.error(f"Error in {domain}: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

# --- DOMAIN ROUTES ---

@app.route("/api/medical", methods=["GET", "POST"])
def medical_api():
    return handle_chat_request('medical', 'medical_history', 'latest_medical_response', 'medical')

@app.route("/api/medical/clear", methods=["POST"])
def medical_clear():
    session.pop('medical_history', None)
    session.pop('current_medical_document', None)
    return jsonify({"status": "cleared"})

@app.route("/api/islamic", methods=["GET", "POST"])
def islamic_api():
    return handle_chat_request('islamic', 'islamic_history', 'latest_islamic_response', 'islamic')

@app.route("/api/islamic/clear", methods=["POST"])
def islamic_clear():
    session.pop('islamic_history', None)
    return jsonify({"status": "cleared"})

@app.route("/api/insurance", methods=["GET", "POST"])
def insurance_api():
    return handle_chat_request('insurance', 'insurance_history', 'latest_insurance_response', 'insurance')

@app.route("/api/insurance/clear", methods=["POST"])
def insurance_clear():
    session.pop('insurance_history', None)
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)