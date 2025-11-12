# 🛍️ Ecommerce Chatbot

An intelligent **E-commerce Data Analysis and Visualization Chatbot** built using **Streamlit**, **Google Gemini 2.5 Flash (via LangChain)**, and **DuckDB**.  
This chatbot allows natural language interaction with structured E-commerce datasets, providing **insights, SQL-based analysis**, and **auto-generated visualizations**.

---
## How to Open / Run the Project

Follow the steps below to install, configure, and launch the **E-Commerce Chatbot AI** on your local machine.

```bash
### 1️ Clone the Repository
git clone https://github.com/Ashuytosh/Ecommerce-Chatbot-AI.git
cd Ecommerce-Chatbot-AI

```
```bash
### 2️ Create and Activate a Virtual Environment
### For Windows:
python -m venv .venv
.venv\Scripts\activate

```

```bash
### For macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
```

```bash
### 3️ Install Required Dependencies
pip install -r requirements.txt

```

```bash
### 4️ Create a .env File (for API Keys)
### Inside the project folder, create a file named .env and add:
GOOGLE_API_KEY=your_basic_chat_api_key
GOOGLE_API_KEY_VIZ=your_visualization_api_key

```

```bash
### 5 Run the Streamlit App
streamlit run app.py

```

## 📊 Dataset Overview

The project leverages **9 datasets** from the Brazilian E-commerce public dataset, providing a rich base for analytics:

| File Name | Description |
|------------|-------------|
| `olist_customers_dataset.csv` | Contains unique customer IDs and their locations |
| `olist_geolocation_dataset.csv` | Provides ZIP code-level geolocation data |
| `olist_orders_dataset.csv` | Main order table containing order metadata and timestamps |
| `olist_order_items_dataset.csv` | Item-level purchase details for each order |
| `olist_order_payments_dataset.csv` | Payment transactions associated with each order |
| `olist_order_reviews_dataset.csv` | Customer review data (ratings, comments, timestamps) |
| `olist_products_dataset.csv` | Product catalog with features like size, weight, and category |
| `olist_sellers_dataset.csv` | Seller identification and location information |
| `product_category_name_translation.csv` | English translations of product categories |

All of these CSVs are stored inside the `Dataset/` folder and connected through a **DuckDB** database (`ecommerce_clean.duckdb`).

---

## 🚀 Features

### 💬 **1. Basic Chat for Data Analysis**
- Ask natural questions like:
  > "List the top 5 product categories by total sales."
- The chatbot translates them into SQL queries automatically and provides factual, concise results.
- Follows the **ReAct reasoning process**:  
  *Thought → Action → Observation → Final Answer*

---

### 📈 **2. Visualization Chat**
- Ask visual questions like:
  > "Show total sales by category as a bar chart."
- Generates visual outputs using the **QuickChart API**.
- Supports multiple chart types:
  - Bar, Line, Pie, Area, Donut, Histogram, Radar, Polar.
- Provides an auto-generated **3-line analytical summary** for every chart.

---

### 🧠 **3. Conversational Memory**
- Implements **LangChain’s `ConversationBufferMemory`** + **`SQLChatMessageHistory`** for long-term chat recall.
- Remembers context and previous questions — enabling **follow-up queries**:
  > "From those same categories, show their average order value."

---

### 🗂️ **4. Past Chat Management**
- Sidebar dropdown shows the last 10 user queries.
- Click any past question → instantly scrolls to that message in the chat window with a **glowing highlight**.
- Option to **clear past chats** from dropdown (memory remains intact in the database).

---

### 💡 **5. Intelligent Summarization**
- Each assistant response includes a **2-line summary** for quick insights.
- Example:
  > 💡 *Insight: Revenue increased steadily across 2018, peaking in Q4.*

---

### 🎨 **6. Streamlit UI Enhancements**
- Custom CSS theme with gradient sidebar, glowing message effect, and modern typography.
- Two chat modes:
  - 🧠 **Basic Chat**
  - 📊 **Visualization Chat**
- Smooth scrolling, blinking "Thinking..." animation, and dynamic feedback.

---

## ⚙️ Workflow

Here’s the detailed **end-to-end architecture** of the chatbot:

1. **Data Preparation**
   - 9 raw CSV files are placed in the `Dataset/` directory.
   - These are cleaned and integrated into **DuckDB**, forming an analytical database.

2. **Database Connection**
   - The app connects to DuckDB using LangChain’s `SQLDatabase`.
   - Provides access to tables for SQL-based data analysis.

3. **LLM Integration**
   - Uses **Google Gemini 2.5 Flash** (via `langchain_google_genai`) as the reasoning model.
   - Model temperature: `0.2` (focused, factual outputs).

4. **Agent Creation**
   - Two agents are created:
     - `setup_agent_basic()` → for text-based analytical queries.
     - `setup_agent_viz()` → for visualization-based queries.
   - Both use **LangChain SQL Agent Toolkit** and follow the **ReAct reasoning loop**.

5. **Memory System**
   - Each agent is linked to:
     - `ConversationBufferMemory` for in-session chat tracking.
     - `SQLChatMessageHistory` for persistent message storage in `Dataset/chat_memory.db`.
   - Ensures continuity even across restarts.

6. **Past Chats & Navigation**
   - Stores up to 10 recent queries in a dropdown.
   - On selection, highlights and scrolls to the message in context.

7. **Visualization Engine**
   - Extracts numeric results as `<Label>: <Value>` pairs.
   - Builds a QuickChart-compatible JSON config.
   - Embeds rendered charts directly in Streamlit.

8. **Insight Summarization**
   - For each visualization or query result, the model generates a concise analytical summary (2–3 lines).

9. **UI Rendering**
   - Streamlit displays chat messages dynamically (user/assistant).
   - Sidebar controls for mode switching, dataset preview, and chat management.

---

## 🧩 Technologies Used

| Category | Technology |
|-----------|-------------|
| **Frontend/UI** | Streamlit, Custom CSS |
| **LLM/AI** | Google Gemini 2.5 Flash via LangChain |
| **Database** | DuckDB |
| **Frameworks** | LangChain, LangChain Community Toolkit |
| **Visualization** | QuickChart API |
| **Memory** | ConversationBufferMemory, SQLChatMessageHistory |
| **Environment** | dotenv |
| **Language** | Python 3.10+ |

---

## 🖼️ Output Screenshots

*(📸 To be added soon — will include Basic Chat and Visualization outputs)*

---

## 👤 Author

**Ashutosh Sahoo**  
**Department of Computer Science and Engineering**  
**Specialization:** Data Science and Analytics | IIIT Nagpur  
📧 [sahooashutosh792@gmail.com](mailto:sahooashutosh792@gmail.com)

---
