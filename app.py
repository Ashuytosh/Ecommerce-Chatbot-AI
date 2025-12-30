import os
import re
import json
import requests
import duckdb
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# -----------------------------
# LLM
# -----------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

#  BASIC CHAT dependencies
try:
    from langchain_community.utilities import SQLDatabase
except:
    from langchain.sql_database import SQLDatabase

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Visualization memory
from langchain.memory import ConversationBufferMemory

# -----------------------------
# Load Env
# -----------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "hidden_key_basic")
os.environ["GOOGLE_API_KEY_VIZ"] = os.getenv("GOOGLE_API_KEY_VIZ", "hidden_key_basic")

# -----------------------------
# Streamlit UI Setup
# -----------------------------
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
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🛍️ E-Commerce Chatbot</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtext"><i>powered by Gemini 2.5 Flash</i></p>', unsafe_allow_html=True)

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.header("🔹 Navigation")
mode = st.sidebar.radio("Choose Chat Mode", ["🧠 Basic Chat", "📊 Visualization Chat"], index=0)

# -----------------------------
# Initialize memory tracking
# -----------------------------
if "chat_history_basic" not in st.session_state:
    st.session_state.chat_history_basic = []

if "chat_history_viz" not in st.session_state:
    st.session_state.chat_history_viz = []


# ==========================================================
# ================== BASIC CHAT (REPLACED) ==================
# ==========================================================

# REPLACED WITH optimized_sql_toolcalling.py logic (session-safe)

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# DB for Basic Chat
BASIC_DB_PATH = "D:/Agentic_AI/Dataset/ecommerce_clean.duckdb"

# Helper: list tables (unused now but handy)
def list_tables(path):
    con = duckdb.connect(path, read_only=True)
    t = [x[0] for x in con.execute("SHOW TABLES").fetchall()]
    con.close()
    return t

# ---------------------------
# Initialize once and cache in st.session_state
# ---------------------------
if "basic_initialized" not in st.session_state:

    # single duckdb connection for Basic Chat (read-only to avoid conflicts)
    #st.session_state.basic_duckdb_conn = duckdb.connect(BASIC_DB_PATH, read_only=True)

    # SQLDatabase wrapper (LangChain)
    st.session_state.db_basic = SQLDatabase.from_uri(
        f"duckdb:///{BASIC_DB_PATH}",
        sample_rows_in_table_info=1,
        include_tables=None
    )

    # LLM
    st.session_state.llm_basic = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.15)

    # Memory (ConversationSummaryBufferMemory kept as your chosen memory)
    st.session_state.memory_basic = ConversationSummaryBufferMemory(
        llm=st.session_state.llm_basic,
        max_token_limit=1000,
        memory_key="chat_history",
        return_messages=True
    )

    # Structured output parser schemas
    response_schemas = [
        ResponseSchema(
            name="RESULT",
            description=(
                "Plain text answer. May contain multiple sentences or lines. "
                "Can include numeric values, percentages, names, explanations, or short paragraphs. "
                "Do NOT include markdown or code formatting."
            )
        ),
        ResponseSchema(
            name="INSIGHT",
            description=(
                "Short interpretation (1–3 lines). "
                "Must be plain text, no markdown, no lists."
            )
        )
    ]
    st.session_state.parser = StructuredOutputParser.from_response_schemas(response_schemas)
    st.session_state.format_instructions = st.session_state.parser.get_format_instructions()

    # System message (keeps your strict format + memory rules can be appended here)
    st.session_state.SYSTEM_MESSAGE_BASIC = f"""
You are a precise SQL analysis assistant connected to a DuckDB database.

STRICT FORMAT REQUIREMENTS:
You MUST follow this exact schema:

{st.session_state.format_instructions}

Example:

RESULT:
sports_leisure has the highest refund rate among the top 5 categories, at approximately 0.61%.

INSIGHT:
sports_leisure stands out due to a higher-than-average rate of refunds.
Customers in this category may have more dissatisfaction or returns.
"""

    # Create SQL Agent (tool-calling)
    st.session_state.agent_basic = create_sql_agent(
        llm=st.session_state.llm_basic,
        toolkit=SQLDatabaseToolkit(db=st.session_state.db_basic, llm=st.session_state.llm_basic),
        agent_type="tool-calling",
        verbose=False,
        memory=st.session_state.memory_basic,
        system_message=st.session_state.SYSTEM_MESSAGE_BASIC,
        prefix=""
    )

    st.session_state.basic_initialized = True

# ---------------------------
# ask_brain_basic: uses the cached objects
# ---------------------------
def ask_brain_basic(query: str):
    """Executes user query → LLM → SQL tool → structured output with retry."""

    # Use cached objects from session_state
    agent_basic = st.session_state.agent_basic
    memory_basic = st.session_state.memory_basic
    llm_basic = st.session_state.llm_basic
    parser = st.session_state.parser
    format_instructions = st.session_state.format_instructions

    # === STEP 0: Build short conversation context ===
    # NOTE: pass the input key required by some memory implementations
    past = memory_basic.load_memory_variables({"input": query}) or {}
    summarized = ""

    if "chat_history" in past and past["chat_history"]:
        for m in past["chat_history"][-6:]:
            # m should be a message object/dict when return_messages=True
            r = getattr(m, "type", None) or getattr(m, "role", "USER")
            c = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else str(m))
            summarized += f"{r.upper()}: {c}\n"

    # === Full contextual prompt (NO formatting rules here) ===
    full_prompt = f"""
Conversation so far:
{summarized}

User question:
{query}
"""

    # === STEP 1: Invoke SQL Agent ===
    try:
        response = agent_basic.invoke({"input": full_prompt})
        text = response["output"] if isinstance(response, dict) else str(response)

    except Exception as e:
        return f"❌ LLM/Agent Error: {e}"

    # === STEP 2: Try to parse into RESULT / INSIGHT ===
    try:
        parsed = parser.parse(text)
        result_block = parsed["RESULT"].strip()
        insight_block = parsed["INSIGHT"].strip()

    except Exception:
        # === STEP 2B: Ask model to reformat ===
        repair_prompt = f"""
Your previous output did NOT follow the required RESULT / INSIGHT schema.

Here is the output you produced:

{text}

Please REFORMAT ONLY — without changing the meaning — following this schema:

{format_instructions}
"""

        try:
            repaired = llm_basic.invoke(repair_prompt).content
            parsed = parser.parse(repaired)

            result_block = parsed["RESULT"].strip()
            insight_block = parsed["INSIGHT"].strip()

        except Exception:
            # === STEP 2C: Final regex fallback ===
            match = re.search(
                r"RESULT:\s*(.*?)\s*INSIGHT:\s*(.*)$",
                text,
                flags=re.DOTALL | re.IGNORECASE
            )

            if not match:
                return text + "\n⚠️ Could not parse output into RESULT / INSIGHT format."

            result_block = "\n".join(
                [ln.strip() for ln in match.group(1).splitlines() if ln.strip()]
            )
            insight_block = match.group(2).strip()

    # === STEP 3: Final combined formatted message ===
    final_answer = f"{result_block}\n\n💡 Insight:\n{insight_block}"

    # === Save memory ===
    memory_basic.save_context({"input": query}, {"output": final_answer})

    return final_answer





# ===============================================================
# ================== VISUALIZATION CHAT (REPLACED) =================
# ===============================================================
# REPLACED WITH YOUR stable final_sql_to_chart_pipeline_all_charts_v2.py


#Automatic SQL generation via Gemini SQL Agent
#SQL output is converted to clean JSON by Gemini (NOT the agent)
#No ReAct parsing issues (agent only runs SQL)
#Universal Chart Engine for all chart types
#QuickChart visualization
#Automatic chart-type detection
#ask_user() wrapper — user writes natural language only
#Fully avoids OutputParsingError
#Pydantic schema validation of numeric data


# =====================================================
# ================== IMPORTS ==========================
# =====================================================
import os
import json
import re
import textwrap
import urllib.parse
from typing import List, Optional, Any, Dict

import requests
from IPython.display import Image, display
import duckdb

from pydantic import BaseModel, ValidationError

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_community.utilities import SQLDatabase
except ImportError:
    from langchain.sql_database import SQLDatabase

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferMemory


# =====================================================
# ================== API KEY ==========================
# =====================================================
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "hidden_key_basic") # from .env


# =====================================================
# ================== DB PATH ==========================
# =====================================================
DB_PATH = r"D:\Agentic_AI\Dataset\ecommerce_file.duckdb"
QUICKCHART_BASE = "https://quickchart.io/chart"


# =====================================================
# =============== PYDANTIC SCHEMAS ====================
# =====================================================
class BasicChartSchema(BaseModel):
    labels: List[str]
    values: List[float]

class ScatterPoint(BaseModel):
    x: float
    y: float

class ScatterChartSchema(BaseModel):
    points: List[ScatterPoint]

class BubblePoint(BaseModel):
    x: float
    y: float
    r: float

class BubbleChartSchema(BaseModel):
    points: List[BubblePoint]

class HistogramSchema(BaseModel):
    values: List[float]


CHART_SCHEMA_MAP = {
    "bar": BasicChartSchema,
    "line": BasicChartSchema,
    "area": BasicChartSchema,
    "pie": BasicChartSchema,
    "doughnut": BasicChartSchema,
    "polararea": BasicChartSchema,
    "radar": BasicChartSchema,
    "scatter": ScatterChartSchema,
    "bubble": BubbleChartSchema,
    "histogram": HistogramSchema,
}



# ============= JSON EXTRACTOR ========================

JSON_PATTERN = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])")

def extract_json_from_text(text: str):
    m = JSON_PATTERN.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except:
            return None


# ============= CHART ENGINE ==========================

def build_chart_config(chart_type: str,
                    labels=None,
                    values=None,
                    points=None,
                    title=None):

    ct = chart_type.lower()
    cfg = {"type": ct}

    if ct in ("bar", "area", "radar"):
        cfg["type"] = "line" if ct in ("line", "area") else "bar" if ct=="bar" else ct
        dataset = {"label": "Series", "data": values or []}
        if ct == "area":
            dataset["fill"] = True
        cfg["data"] = {"labels": labels or [], "datasets": [dataset]}
        
    # LINE CHART (separate)
    
    if ct == "line":
        cfg["type"] = "line"
        dataset = {
            "label": "Series",
            "data": values or [],
            "fill": False   
        }
        cfg["data"] = {"labels": labels or [], "datasets": [dataset]}
    

    elif ct in ("pie", "doughnut", "polararea"):
        cfg["type"] = "pie" if ct=="pie" else "doughnut" if ct=="doughnut" else "polarArea"
        cfg["data"] = {"labels": labels or [], "datasets": [{"data": values or []}]}

    elif ct == "scatter":
        cfg = {
            "type": "scatter",
            "data": {"datasets": [{"label": "Scatter", "data": points or []}]}
        }

    elif ct == "bubble":
        cfg = {
            "type": "bubble",
            "data": {"datasets": [{"label": "Bubble", "data": points or []}]}
        }

    elif ct == "histogram":
        cfg = {
            "type": "bar",
            "data": {
                "labels": [str(v) for v in values],
                "datasets": [{"label": "Histogram", "data": values}]
            }
        }

    if title:
        cfg["options"] = {"plugins": {"title": {"display": True, "text": title}}}

    return cfg


def quickchart_url_from_config(cfg, width=900, height=450):
    return QUICKCHART_BASE + "?" + urllib.parse.urlencode({
        "c": json.dumps(cfg, separators=(",", ":")),
        "w": width,
        "h": height
    })



# ================ AGENT SETUP ========================

base_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

db = SQLDatabase.from_uri(f"duckdb:///{DB_PATH}", sample_rows_in_table_info=1)
toolkit = SQLDatabaseToolkit(db=db, llm=base_llm)

CUSTOM_PREFIX = """
You are a SQL Agent that ONLY generates and runs SQL queries.
Do NOT return JSON.
Do NOT return analysis.
Do NOT return explanations.
Just return the SQL result as plain text table or values.

If the user is asking for ANY chart, graph, visualization, trend or distribution:
ALWAYS use the SQL tool first to fetch the data.
NEVER answer directly.

AVAILABLE TABLES AND COLUMNS (USE THESE EXACT NAMES):

1. customers(
    customer_id, customer_unique_id, customer_zip_code_prefix,
    customer_city, customer_state
)

2. geolocation(
    geolocation_zip_code_prefix, geolocation_lat, geolocation_lng,
    geolocation_city, geolocation_state
)

3. orders(
    order_id, customer_id, order_status, order_purchase_timestamp,
    order_approved_at, order_delivered_carrier_date,
    order_delivered_customer_date, order_estimated_delivery_date
)

4. order_items(
    order_id, order_item_id, product_id, seller_id,
    shipping_limit_date, price, freight_value
)

5. order_payments(
    order_id, payment_sequential, payment_type,
    payment_installments, payment_value
)

6. order_reviews(
    review_id, order_id, review_score,
    review_creation_date, review_answer_timestamp
)

7. products(
    product_id, product_category_name, product_name_lenght,
    product_description_lenght, product_photos_qty,
    product_weight_g, product_length_cm, product_height_cm,
    product_width_cm
)

8. sellers(
    seller_id, seller_zip_code_prefix, seller_city, seller_state
)

9. p_category_name_translation(
    product_category_name, product_category_name_english
)

"""

sql_agent = create_sql_agent(
    llm=base_llm,
    toolkit=toolkit,
    verbose=False,
    handle_parsing_errors=True,
    memory=memory,
    agent_type="tool-calling",
    prefix=CUSTOM_PREFIX
)


print("🔥 System Loaded: SQL Agent + JSON Converter + Universal Chart Engine Ready")



# ========= AUTOMATIC CHART-TYPE DETECTOR =============

def detect_chart_type(question: str) -> str:
    q = question.lower()

    if any(w in q for w in ["line", "trend", "monthly", "over time", "time series"]):
        return "line"
    if "bar" in q:
        return "bar"
    if "pie" in q:
        return "pie"
    if "donut" in q or "doughnut" in q:
        return "doughnut"
    if "scatter" in q:
        return "scatter"
    if "bubble" in q:
        return "bubble"
    if "histogram" in q or "frequency" in q:
        return "histogram"
    if "area" in q:
        return "area"
    if "polar" in q:
        return "polarArea"

    return "bar"



# ******* MAIN PIPELINE ************

def ask_pipeline(user_question: str,
                chart_type: str,
                title: Optional[str] = None,
                width=900,
                height=450):

    # --------------------------------------------------
    # STEP 1 — RUN SQL via agent (NO JSON here)
    # --------------------------------------------------
    sql_result = sql_agent.invoke({"input": user_question})
    raw_sql_output = sql_result.get("output", str(sql_result))

    if not raw_sql_output.strip():
        return {"error": "SQL returned empty result"}

    # --------------------------------------------------
    # STEP 2 — Ask Gemini to convert SQL → JSON
    # --------------------------------------------------
    schema_cls = CHART_SCHEMA_MAP.get(chart_type.lower(), BasicChartSchema)

    json_prompt = f"""
Convert the following SQL result into pure JSON that matches this schema:
{schema_cls.schema_json(indent=2)}

SQL RESULT:
{raw_sql_output}

Rules:
- Output ONLY JSON
- No markdown, no ```json fences
- Labels must be strings
- Values must be numeric
"""

    json_raw = base_llm.predict(json_prompt)
    json_data = extract_json_from_text(json_raw)

    if json_data is None:
        return {"error": "Could not extract JSON", "llm_output": json_raw}

    # --------------------------------------------------
    # STEP 3 — Validate with Pydantic
    # --------------------------------------------------
    try:
        schema_instance = schema_cls.parse_obj(json_data)
    except ValidationError as ve:
        return {"error": "Schema validation failed", "detail": ve.errors(), "json": json_data}

    # --------------------------------------------------
    # STEP 4 — Build Chart JSON
    # --------------------------------------------------
    if isinstance(schema_instance, BasicChartSchema):
        cfg = build_chart_config(chart_type, schema_instance.labels, schema_instance.values, title=title)
    elif isinstance(schema_instance, HistogramSchema):
        cfg = build_chart_config("histogram", values=schema_instance.values, title=title)
    elif isinstance(schema_instance, ScatterChartSchema):
        pts = [{"x": p.x, "y": p.y} for p in schema_instance.points]
        cfg = build_chart_config("scatter", points=pts, title=title)
    elif isinstance(schema_instance, BubbleChartSchema):
        pts = [{"x": p.x, "y": p.y, "r": p.r} for p in schema_instance.points]
        cfg = build_chart_config("bubble", points=pts, title=title)

    # --------------------------------------------------
    # STEP 5 — Render Chart
    # --------------------------------------------------
    chart_url = quickchart_url_from_config(cfg, width, height)

    # --------------------------------------------------
    # STEP 6 — Summary
    # --------------------------------------------------
    try:
        summary = base_llm.predict(f"Summarize these results in 2-3 lines: {json_data}")
    except:
        summary = "Summary unavailable."

    # Return data for caller (Streamlit UI will render)
    return {
        "chart_url": chart_url,
        "summary": summary,
        "sql_output": raw_sql_output,
        "json_output": json_data
    }


# =====================================================
# ============== USER-FRIENDLY WRAPPER ================
# =====================================================
def ask_user(question: str):
    chart_type = detect_chart_type(question)
    return ask_pipeline(
        user_question=question,
        chart_type=chart_type,
        title=question
    )


# Backwards-compatible wrapper used by Streamlit UI
def ask_visual(question: str):
    out = ask_user(question)
    if isinstance(out, dict) and out.get("chart_url"):
        # Compose Markdown with image + summary for Streamlit
        md = f"![]({out['chart_url']})\n\n💡 **Insight:**\n{out['summary']}"
        return md
    elif isinstance(out, dict) and out.get("error"):
        return f"⚠️ Error: {out.get('error')}"
    else:
        return str(out)


print("\n✅ Ready")

# =====================================================
# ====================== BASIC CHAT UI ==========================
if mode == "🧠 Basic Chat":
    
        # === Dataset Dropdown ===
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


    for role, msg in st.session_state.chat_history_basic:
        st.chat_message(role).markdown(msg)

    if prompt := st.chat_input("Ask about sales, trends, revenue..."):

    # show user bubble
        st.chat_message("user").markdown(prompt)

        # ⭐ INSERT THINKING BLOCK HERE
        thinking = st.chat_message("assistant").markdown(
            "<div class='thinking'>Thinking...</div>",
            unsafe_allow_html=True
        )

        # call new logic
        reply = ask_brain_basic(prompt)

        # remove thinking animation
        thinking.empty()

        # show final assistant reply
        st.chat_message("assistant").markdown(reply, unsafe_allow_html=True)

        # store in Streamlit session history
        st.session_state.chat_history_basic.append(("user", prompt))
        st.session_state.chat_history_basic.append(("assistant", reply))



# ===============================================================
# ================== VISUALIZATION CHAT UI ======================
else:

    for role, msg in st.session_state.chat_history_viz:
        st.chat_message(role).markdown(msg)

    if prompt := st.chat_input("Ask for visual insights..."):

        st.chat_message("user").markdown(prompt)

        thinking = st.chat_message("assistant").markdown(
            "<div class='thinking'>Thinking...</div>",
            unsafe_allow_html=True
        )

        reply = ask_visual(prompt)   # compatible wrapper

        thinking.empty()

        st.chat_message("assistant").markdown(reply, unsafe_allow_html=True)

        st.session_state.chat_history_viz.append(("user", prompt))
        st.session_state.chat_history_viz.append(("assistant", reply))
