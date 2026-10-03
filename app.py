"""
app.py
Stock Saathi: Inventory / Stock Query Assistant for a small retailer (Use Case #11).

Run locally (virtual environment active, from the project folder):
    streamlit run app.py

Architecture:
    user question -> Gemini decides which tool to call -> tools.py reads data/store.db and calculates
    -> Gemini phrases the answer from the tool result. Gemini never invents numbers.
"""
import json
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

import tools
from prompts import build_system_prompt

PRIMARY_MODEL_DEFAULT = "gemini-flash-latest"            # always points to Google's newest Flash model
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]   # tried in order if the current model fails
MAX_INPUT_CHARS = 500
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

STARTER_QUESTIONS = [
    "What's running low today?",
    "Tata namak kitna bacha hai?",
    "What should I reorder today?",
    "How is Red Bull selling?",
    "Do we have Surf?",
    "What's not selling?",
]

st.set_page_config(page_title="Stock Saathi | Inventory Assistant", page_icon="📦", layout="wide")


# --------------------------------------------------------------------------------------
# Setup: data, API key, Gemini chat
# --------------------------------------------------------------------------------------
overview = tools.get_store_overview()
STORE_NAME, AS_OF = overview["store_name"], overview["data_as_of"]


def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.environ.get(name, default)


API_KEY = get_secret("GEMINI_API_KEY")
PRIMARY_MODEL = get_secret("GEMINI_MODEL", PRIMARY_MODEL_DEFAULT)


def make_config():
    return types.GenerateContentConfig(
        system_instruction=build_system_prompt(STORE_NAME, AS_OF, overview["categories"]),
        tools=tools.TOOLS,              # Python functions: the SDK calls them automatically
        temperature=0.2,                # low temperature = consistent, factual answers
    )


def new_chat(model, history=None):
    return st.session_state.client.chats.create(model=model, config=make_config(), history=history or [])


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []          # what we display: {role, content, tools}
    if API_KEY and "client" not in st.session_state:
        st.session_state.client = genai.Client(api_key=API_KEY)
    if API_KEY and "chat" not in st.session_state:
        st.session_state.model = PRIMARY_MODEL
        st.session_state.chat = new_chat(PRIMARY_MODEL)


def tool_calls_from(response):
    """List the tools Gemini called for the latest question (for the 'Data used' panel)."""
    history = getattr(response, "automatic_function_calling_history", None) or []
    # start after the last user message that contains text (the current question)
    start = 0
    for i, content in enumerate(history):
        if content.role == "user" and any(getattr(p, "text", None) for p in (content.parts or [])):
            start = i
    calls = []
    for content in history[start:]:
        for p in content.parts or []:
            fc = getattr(p, "function_call", None)
            if fc:
                calls.append({"tool": fc.name, "args": dict(fc.args or {})})
    return calls


def ask_gemini(question):
    """Send a question. Retries server errors once, then falls back to other models.

    Returns (answer_text, tool_calls, error_detail_or_None).
    """
    current = st.session_state.model
    models = [current] + [m for m in FALLBACK_MODELS if m != current]
    last_err = None
    for model in models:
        if model != st.session_state.model:
            # switch model but keep the conversation so far
            history = st.session_state.chat.get_history()
            st.session_state.model = model
            st.session_state.chat = new_chat(model, history)
        for attempt in range(2):
            try:
                resp = st.session_state.chat.send_message(question)
                text = resp.text or "Sorry, I could not form an answer. Please rephrase."
                return text, tool_calls_from(resp), None
            except Exception as e:
                last_err = e
                code = getattr(e, "code", None)
                if code in (401, 403):
                    return ("The Gemini API key was rejected. Please check GEMINI_API_KEY in "
                            ".streamlit/secrets.toml.", [], f"{type(e).__name__} {code}: {str(e)[:300]}")
                if code in (500, 502, 503, 504) and attempt == 0:
                    time.sleep(2)          # temporary server problem: wait and retry the same model
                    continue
                break                      # 404/400/429/other: move on to the next model

    code = getattr(last_err, "code", None)
    if code == 429:
        msg = "The free-tier request limit has been reached. Please wait a minute and try again."
    elif code in (500, 502, 503, 504):
        msg = "Google's AI servers are busy right now. Please try again in a moment."
    else:
        msg = "I couldn't reach the AI service."
    return (msg + " Meanwhile, the **Dashboard** tab works without AI.", [],
            f"{type(last_err).__name__} {code}: {str(last_err)[:300]}")


def log_turn(question, answer, calls, error=None):
    """Save each turn to logs/chat_log.jsonl: useful evidence for testing in the report."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "chat_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                "model": st.session_state.get("model"), "question": question,
                                "tools_called": calls, "answer": answer, "error": error}, ensure_ascii=False) + "\n")
    except Exception:
        pass                                   # logging must never break the app


init_state()

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("📦 Stock Saathi")
    st.caption(f"{STORE_NAME}  \nData as of **{AS_OF}**")
    c = overview["status_counts"]
    col1, col2 = st.columns(2)
    col1.metric("Products", overview["total_products"])
    col2.metric("Out of stock", c.get("OUT_OF_STOCK", 0))
    col1.metric("Low stock", c.get("LOW_STOCK", 0))
    col2.metric("Dead stock", c.get("DEAD_STOCK", 0))
    st.metric("Stock value (at cost)", f"₹{overview['stock_value_at_cost_inr']:,.0f}")
    st.divider()
    if st.button("🗑️ Clear chat", width="stretch"):
        st.session_state.messages = []
        if API_KEY:
            st.session_state.chat = new_chat(st.session_state.model)
        st.rerun()
    st.caption(f"Model: `{st.session_state.get('model', 'not connected')}`")
    st.info("I'm an AI assistant. I only read stock data and suggest reorders; "
            "the owner makes the final decision. Double-check physical stock before large orders.", icon="ℹ️")

# --------------------------------------------------------------------------------------
# Main area: Chat tab and Dashboard tab
# --------------------------------------------------------------------------------------
st.title("Stock Saathi: Inventory Assistant")
tab_chat, tab_dash = st.tabs(["💬 Chat", "📊 Dashboard"])

with tab_chat:
    if not API_KEY:
        st.error("GEMINI_API_KEY not found. Add it to .streamlit/secrets.toml (see README). "
                 "The Dashboard tab still works.")
    else:
        chat_area = st.container()          # messages go here, ABOVE the input box

        def show_tools(calls):
            if calls:
                with st.expander("Data used"):
                    for call in calls:
                        st.code(f"{call['tool']}({json.dumps(call['args'], ensure_ascii=False)})", language="python")

        with chat_area:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📦"):
                    st.markdown(msg["content"])
                    show_tools(msg.get("tools"))
                    if msg.get("error"):
                        with st.expander("Technical details"):
                            st.code(msg["error"])

            # starter buttons on an empty chat
            if not st.session_state.messages:
                st.markdown("Ask about stock in English, Hindi or Hinglish. Try one of these:")
                cols = st.columns(3)
                for i, q in enumerate(STARTER_QUESTIONS):
                    if cols[i % 3].button(q, key=f"starter_{i}", width="stretch"):
                        st.session_state.pending = q
                        st.rerun()

        typed = st.chat_input("Ask about stock, e.g. 'atta 5kg kitna hai?'")
        question = typed or st.session_state.pop("pending", None)

        if question:
            question = question.strip()[:MAX_INPUT_CHARS]
            st.session_state.messages.append({"role": "user", "content": question})
            with chat_area:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(question)
                with st.chat_message("assistant", avatar="📦"):
                    with st.spinner("Checking stock..."):
                        answer, calls, error = ask_gemini(question)
                    st.markdown(answer)
                    show_tools(calls)
                    if error:
                        with st.expander("Technical details"):
                            st.code(error)
            st.session_state.messages.append({"role": "assistant", "content": answer, "tools": calls, "error": error})
            log_turn(question, answer, calls, error)
            st.rerun()                      # redraw so the starter buttons disappear

with tab_dash:
    st.caption(f"Calculated directly from the store database (no AI). Data as of {AS_OF}.")

    st.subheader("🚨 Stock alerts")
    low = tools.get_low_stock_items()
    if low["items"]:
        df_low = pd.DataFrame(low["items"]).rename(columns={
            "product": "Product", "on_hand": "On hand", "on_order": "On order", "status": "Status",
            "days_of_cover": "Days of cover", "covered_by_pending_order": "Covered by pending order"})
        st.dataframe(df_low.drop(columns=["sku"]), hide_index=True, width="stretch")
    else:
        st.success("No items are low on stock.")

    st.subheader("🛒 Reorder suggestions")
    reo = tools.get_reorder_suggestions()
    rows = [{"Supplier": g["supplier"], "Product": it["product"], "Order qty": it["order_qty"],
             "On hand": it["on_hand"], "Trend": it["trend"], "Reason": it["reason"]}
            for g in reo["suppliers"] for it in g["items"]]
    if rows:
        df_reo = pd.DataFrame(rows)
        st.write(f"**{reo['items_to_order']} items**, estimated value at cost "
                 f"**₹{reo['total_order_value_at_cost_inr']:,.0f}**")
        st.dataframe(df_reo, hide_index=True, width="stretch")
        st.download_button("⬇️ Download reorder list (CSV)", df_reo.to_csv(index=False).encode("utf-8"),
                           file_name=f"reorder_list_{AS_OF}.csv", mime="text/csv")
        below = [g["supplier"] for g in reo["suppliers"] if not g["meets_min_order_value"]]
        if below:
            st.warning("Below supplier minimum order value: " + ", ".join(below) +
                       ". Consider combining with other items from the same supplier.")
    else:
        st.success("Nothing needs reordering today.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🏆 Top sellers (30 days)")
        top = pd.DataFrame(tools.get_top_sellers(days=30, limit=10)["top_products"])
        st.bar_chart(top.set_index("product")["units_sold"], horizontal=True)
    with col_b:
        st.subheader("🐢 Slow-moving stock")
        slow = tools.get_slow_moving_items()
        st.write(f"₹{slow['total_value_tied_up_inr']:,.0f} tied up in {slow['count']} items")
        st.dataframe(pd.DataFrame(slow["items"]).rename(columns={
            "product": "Product", "status": "Status", "on_hand": "On hand", "days_of_cover": "Days of cover",
            "days_since_last_sale": "Days since last sale", "stock_value_at_cost_inr": "Value (₹)"}),
            hide_index=True, width="stretch")
