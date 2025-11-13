# === app.py ===
import os
import re
import json
import time
import sqlite3
import requests
import duckdb
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# === Remove broken memory imports (we use our own SQLite memory) ===

# === Load API Keys ===
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "hidden_key_basic")
os.environ["GOOGLE_API_KEY_VIZ"] = os.getenv("GOOGLE_API_KEY_VIZ", "hidden_key_basics")

# === Streamlit Setup ===
st.set_page_config(page_title="E-Commerce Chatbot", page_icon="🛍️", layout="wide")
st.markdown("""
<style>
body { background-color: #f9fbfd; font-family: 'Inter', sans-serif; }
.sidebar .sidebar-content { background: linear-gradient(180deg, #e3f2fd 0%, #b3e5fc 100%); }
h1, h3 { color: #1976d2; text-align: center; font-weight: 700; }
.subtext { color: #0288d1; font-size: 16px; text-align: center; margin-top: -10px; }
.summary-text { color: #0277bd; font-style: italic; font-weight: 600; }
.thinking { color: #0288d1; font-style: italic; font-weight: 500; animation: blink 1s infinite; }
@keyframes blink { 0%{opacity:0.2;} 50%{opacity:1;} 100%{opacity:0.2;} }
.glow { animation: glowEffect 0.5s ease-in-out; border-radius: 10px; background-color: #fff3cd; }
@keyframes glowEffect { 0% { box-shadow: 0 0 0px #ffc107; } 50% { box-shadow: 0 0 25px #ffc107; } 100% { box-shadow: 0 0 0px #ffc107; } }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🛍️ E-Commerce Chatbot</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtext"><i>powered by Gemini 2.5 Flash</i></p>', unsafe_allow_html=True)

# === Sidebar Navigation ===
st.sidebar.header("🔹 Navigation")
mode = st.sidebar.radio("Choose Chat Mode", ["🧠 Basic Chat", "📊 Visualization Chat"], index=0)

# === Helper: Scroll to element ===
def scroll_to_message(message_id):
    js = f"""
    <script>
    function scrollAndHold(targetId) {{
        const target = document.getElementById(targetId);
        if (target) {{
            target.scrollIntoView({{ behavior: 'smooth', block: 'center', inline: 'nearest' }});
            target.classList.add('glow');
            setTimeout(() => target.classList.remove('glow'), 1200);
            let lastScroll = window.scrollY;
            const lock = setInterval(() => {{
                if (Math.abs(window.scrollY - lastScroll) > 50) {{
                    clearInterval(lock);
                }} else {{
                    window.scrollTo({{ top: target.offsetTop - window.innerHeight / 2 }});
                }}
                lastScroll = window.scrollY;
            }}, 300);
            setTimeout(() => clearInterval(lock), 2000);
        }}
    }}
    setTimeout(() => scrollAndHold("{message_id}"), 1200);
    </script>
    """
    st.components.v1.html(js, height=0)

# === Persistent Past Chats in session_state ===
if "past_chat_dropdown" not in st.session_state:
    st.session_state.past_chat_dropdown = []
def update_past_chats(question):
    if question not in st.session_state.past_chat_dropdown:
        st.session_state.past_chat_dropdown.insert(0, question)
        if len(st.session_state.past_chat_dropdown) > 10:
            st.session_state.past_chat_dropdown.pop()
def clear_past_chats():
    st.session_state.past_chat_dropdown = []

# === Simple SQLite-backed chat memory (works on Streamlit Cloud) ===
MEM_DB_PATH = "Dataset/chat_memory.db"
def ensure_memory_db():
    os.makedirs(os.path.dirname(MEM_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(MEM_DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        ts REAL
    )
    """)
    conn.commit()
    return conn

_mem_conn = ensure_memory_db()

def save_message(session_id: str, role: str, content: str):
    cur = _mem_conn.cursor()
    cur.execute("INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()))
    _mem_conn.commit()

def load_recent_messages(session_id: str, limit: int = 10):
    cur = _mem_conn.cursor()
    cur.execute("SELECT role, content, ts FROM messages WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
                (session_id, limit))
    rows = cur.fetchall()
    # return newest-last -> oldest-first
    rows.reverse()
    return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]

# === Setup Agent Functions (we DO NOT pass a LangChain Memory object) ===
def setup_agent_basic(db_path):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "hidden_key_basic")
    db = SQLDatabase.from_uri(f"duckdb:///{db_path}", sample_rows_in_table_info=1)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    prefix = """
You are a reasoning SQL analysis agent connected to a DuckDB database.
Always follow the ReAct reasoning process.
The final output must begin with 'Final Answer:' followed by plain text.
"""
    # Note: we intentionally do not pass a LangChain memory object; we manage context externally.
    agent = create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=False,
        handle_parsing_errors=True,
        agent_type="zero-shot-react-description", prefix=prefix
    )
    return agent

def setup_agent_viz(db_path):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY_VIZ", "hidden_key_basics")
    db = SQLDatabase.from_uri(f"duckdb:///{db_path}", sample_rows_in_table_info=1)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    prefix = """
You are a reasoning SQL visualization agent connected to a DuckDB database.
Always follow ReAct reasoning structure. Final output must begin with 'Final Answer:'.
"""
    agent = create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=False,
        handle_parsing_errors=True,
        agent_type="zero-shot-react-description", prefix=prefix
    )
    return agent

# ===============================
# ===== BASIC CHAT SECTION =====
# ===============================
if mode == "🧠 Basic Chat":
    st.markdown("### 💬 Ask your data analysis questions below")
    agent_basic = setup_agent_basic("D:/Agentic_AI/Dataset/ecommerce_clean.duckdb")
    session_id = "user_1"

    # Dataset dropdown
    st.sidebar.markdown("### 🧾 Dataset Overview")
    datasets = {
        "Customers Table": "Dataset/olist_customers_dataset.csv",
        "Orders Table": "Dataset/olist_orders_dataset.csv",
        "Order Items Table": "Dataset/olist_order_items_dataset.csv",
        "Order Payments Table": "Dataset/olist_order_payments_dataset.csv",
        "Order Review Table": "Dataset/olist_order_reviews_dataset.csv",
        "Products Table": "Dataset/olist_products_dataset.csv",
        "Sellers Table": "Dataset/olist_sellers_dataset.csv",
        "Geolocation Table": "Dataset/olist_geolocation_dataset.csv",
        "Translation Table": "Dataset/product_category_name_translation.csv"
    }
    selected_dataset = st.sidebar.selectbox("Select table to preview:", ["None"] + list(datasets.keys()))
    if selected_dataset != "None":
        try:
            df = pd.read_csv(datasets[selected_dataset])
            st.dataframe(df.head(20))
        except Exception as e:
            st.error(f"⚠️ Could not load {selected_dataset}: {e}")

    # Past chats
    st.sidebar.markdown("### 💬 Past Chats")
    selected_past = st.sidebar.selectbox("Select previous question:", ["None"] + st.session_state.past_chat_dropdown, key="basic_past")
    if st.sidebar.button("🧹 Clear Past Chats", key="clear_basic"):
        clear_past_chats()
        st.sidebar.success("Cleared sidebar history (DB memory remains intact).")

    # Summarizer helper
    def summarize_output(text):
        try:
            summarizer = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)
            resp = summarizer.invoke(f"Summarize this in 2 concise factual lines:\n{text}")
            return f"<div class='summary-text'>{resp.content.strip()}</div>"
        except Exception:
            return "<div class='summary-text'>(No summary generated.)</div>"

    # Memory-aware ask
    def ask_brain(query: str):
        try:
            recent = load_recent_messages(session_id, limit=8)
            memory_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent]) if recent else "No prior conversation."
            contextual_prompt = f"""
You are a precise SQL data analyst AI.

Conversation history:
{memory_context}

User's new question:
{query}

Rules:
- Base your reasoning on prior context if relevant.
- Use SQL only for real tables.
- Keep answers concise and factual.

Please answer and if you run SQL, include queries and outputs in your chain-of-thought, but the FINAL ANSWER should be prefixed with 'Final Answer:'.
"""
            response = agent_basic.invoke({"input": contextual_prompt}, return_intermediate_steps=True)
            answer = response.get("output", str(response))
            cleaned = re.sub(r'^\s*[-•]\s*', '', answer.strip())
            parts = re.split(r'\s*[;,]\s*', cleaned)
            final_text = "\n".join(parts) if len(parts) > 1 else cleaned
            summary = summarize_output(final_text)
            # save messages to our sqlite memory
            save_message(session_id, "user", query)
            save_message(session_id, "assistant", final_text)
            update_past_chats(query)
            return f"{final_text}\n\n💡 Insight:\n{summary}"
        except Exception as e:
            return f"⚠️ Error: {e}"

    # Chat display using session_state lists (UI only)
    if "chat_history_basic" not in st.session_state:
        st.session_state.chat_history_basic = []

    for idx, (role, msg) in enumerate(st.session_state.chat_history_basic):
        msg_id = f"msg_basic_{idx}"
        st.markdown(f"<div id='{msg_id}'></div>", unsafe_allow_html=True)
        st.chat_message(role).markdown(msg, unsafe_allow_html=True)
        if selected_past != "None" and role == "user" and selected_past.strip() == msg.strip():
            scroll_to_message(msg_id)

    if prompt := st.chat_input("Ask about sales, trends, revenue..."):
        st.chat_message("user").markdown(prompt)
        thinking = st.chat_message("assistant").markdown("<div class='thinking'>Thinking...</div>", unsafe_allow_html=True)
        reply = ask_brain(prompt)
        thinking.empty()
        st.chat_message("assistant").markdown(reply, unsafe_allow_html=True)
        st.session_state.chat_history_basic.append(("user", prompt))
        st.session_state.chat_history_basic.append(("assistant", reply))

# ===============================
# === VISUALIZATION CHAT TAB ===
# ===============================
else:
    st.markdown("### 📊 Ask for visualizations and insights")
    agent_viz = setup_agent_viz("D:/Agentic_AI/Dataset/ecommerce_file.duckdb")
    session_id_viz = "user_1_viz"

    st.sidebar.markdown("### 📁 Available Charts")
    st.sidebar.info("Bar, Line, Pie, Area, Histogram, Radar, Donut, Polar")

    st.sidebar.markdown("### 💬 Past Chats")
    selected_past = st.sidebar.selectbox("Select previous question:", ["None"] + st.session_state.past_chat_dropdown, key="viz_past")
    if st.sidebar.button("🧹 Clear Past Chats", key="clear_viz"):
        clear_past_chats()
        st.sidebar.success("Cleared sidebar history (DB memory remains intact).")

    def ask_with_viz(query):
        try:
            # include recent viz memory
            recent = load_recent_messages(session_id_viz, limit=6)
            memory_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent]) if recent else ""
            enforced_prompt = f"""
You are a SQL visualization assistant.
Conversation context:
{memory_context}

User request: {query}

Return ONLY Label: Value pairs (no markdown/bullets). Provide numeric outputs suitable for plotting.
"""
            resp = agent_viz.invoke({"input": enforced_prompt})
            text = resp.get("output", str(resp))
            pairs = re.findall(r"\s*([\w\s&:/\-\(\)\d]+):\s*([\d,.]+)", text)
            if not pairs:
                return "⚠️ No numeric data found."
            labels = [p[0].strip() for p in pairs]
            vals = [float(p[1].replace(',', '')) for p in pairs]
            chart_type = next((t for t in ["bar","pie","line","area","donut","histogram","radar","polar"]
                            if t in query.lower()), "bar")
            base = {"labels": labels, "datasets": [{"label": query, "data": vals}]}
            cfg = {"type": chart_type if chart_type!="area" else "line",
                "data": base, "options": {"plugins": {"title": {"display": True, "text": query}}}}
            if chart_type=="line": cfg["options"]["elements"]={"line":{"fill":False}}
            if chart_type=="donut": cfg["type"]="doughnut"
            if chart_type=="polar": cfg["type"]="polarArea"
            if chart_type=="area": cfg["options"]["elements"]={"line":{"fill":True}}
            chart_url = f"https://quickchart.io/chart?c={requests.utils.quote(json.dumps(cfg))}"
            summary_prompt = f"Summarize these results (3 concise analytical lines):\n{text}"
            summary_resp = agent_viz.invoke({"input": summary_prompt})
            summary_text = summary_resp.get("output", str(summary_resp))
            save_message(session_id_viz, "user", query)
            save_message(session_id_viz, "assistant", summary_text)
            update_past_chats(query)
            return f"🖼️ Chart:\n![]({chart_url})\n\n💡 Insight:\n{summary_text}"
        except Exception as e:
            return f"⚠️ Visualization error: {e}"

    if "chat_history_viz" not in st.session_state:
        st.session_state.chat_history_viz = []

    for idx, (role, msg) in enumerate(st.session_state.chat_history_viz):
        msg_id = f"msg_viz_{idx}"
        st.markdown(f"<div id='{msg_id}'></div>", unsafe_allow_html=True)
        st.chat_message(role).markdown(msg, unsafe_allow_html=True)
        if selected_past != "None" and role == "user" and selected_past.strip() == msg.strip():
            scroll_to_message(msg_id)

    if prompt := st.chat_input("Ask for visual insights (e.g., sales by month as area chart)"):
        st.chat_message("user").markdown(prompt)
        thinking = st.chat_message("assistant").markdown("<div class='thinking'>Thinking...</div>", unsafe_allow_html=True)
        reply = ask_with_viz(prompt)
        thinking.empty()
        st.chat_message("assistant").markdown(reply, unsafe_allow_html=True)
        st.session_state.chat_history_viz.append(("user", prompt))
        st.session_state.chat_history_viz.append(("assistant", reply))
