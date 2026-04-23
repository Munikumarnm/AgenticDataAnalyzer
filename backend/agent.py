"""
agent.py — LangGraph-powered Q&A agent for DataCopilot.

Graph flow:
  interpret  →  generate_sql  →  execute  →  summarize

- interpret    : heuristic layer — builds SQL directly for simple aggregate patterns
                 and sets a chart_hint for the UI
- generate_sql : Gemini LLM fallback for complex/freeform questions
- execute      : runs SQL against SQLite
- summarize    : formats a human-readable answer (single value or LLM summary)
"""

import os
import re
from typing import TypedDict, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from backend.database import get_schema, run_query

load_dotenv()

# ── LLM setup — supports both google-genai (new) and google-generativeai (old) ─
# Read from env first, then fall back to st.secrets (Streamlit Cloud)
_KEY: str = os.getenv("GEMINI_API_KEY", "")
if not _KEY:
    try:
        import streamlit as _st
        _KEY = _st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
_MODEL = "gemini-2.5-flash"
_client = None          # new SDK  (google-genai)
_legacy_model = None    # old SDK  (google-generativeai)

try:
    from google import genai as _new_sdk
    if _KEY:
        _client = _new_sdk.Client(api_key=_KEY)
except ImportError:
    try:
        import google.generativeai as _old_sdk  # type: ignore
        if _KEY:
            _old_sdk.configure(api_key=_KEY)
            _legacy_model = _old_sdk.GenerativeModel(_MODEL)
    except ImportError:
        pass   # No Gemini SDK installed — LLM features disabled


def _llm_generate(prompt: str) -> str:
    """Call whichever Gemini SDK is available and return the response text."""
    if _client is not None:
        resp = _client.models.generate_content(model=_MODEL, contents=prompt)
        return resp.text
    if _legacy_model is not None:
        resp = _legacy_model.generate_content(prompt)
        return resp.text
    raise RuntimeError("No Gemini SDK available. Run: pip install google-genai")


# ── State ──────────────────────────────────────────────────────────────────────

class QAState(TypedDict):
    question:   str
    sql:        str
    result:     pd.DataFrame
    outcome:    str
    chart_hint: str                     # 'bar' | 'line' | 'histogram' | 'pie' | 'scatter' | ''
    status:     Literal["OK", "FAILED"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _quote(name: str) -> str:
    """Double-quote an SQLite identifier, escaping internal double-quotes."""
    return f'"{name.replace(chr(34), chr(34) + chr(34))}"'


def _alias(name: str) -> str:
    """Produce a safe SQL alias from an arbitrary string."""
    a = re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_")
    a = re.sub(r"__+", "_", a)
    return ("col_" + a) if (a and a[0].isdigit()) else (a or "value")


# ── Node 1 — Heuristic intent interpreter ─────────────────────────────────────

def node_interpret(state: QAState) -> QAState:
    """
    Detects simple aggregate patterns (count/sum/avg/min/max) and builds
    SQL directly, bypassing the LLM for speed and reliability.
    Also sets chart_hint based on keywords in the question.
    """
    q = state["question"].strip().lower()
    columns, num_cols = get_schema()

    # ── Detect aggregate operator
    op: Optional[str] = None
    if "how many" in q or ("count" in q and "by" not in q):
        op = "count"
    elif any(x in q for x in ["total", "sum of", "sum "]):
        op = "sum"
    elif any(x in q for x in ["average", "avg", "mean"]):
        op = "avg"
    elif any(x in q for x in ["minimum", "min ", "lowest", "smallest", "least"]):
        op = "min"
    elif any(x in q for x in ["maximum", "max ", "highest", "largest", "most"]):
        op = "max"

    # ── Find explicit column mention
    # Checks both exact name ("price") and space-separated form ("units sold" → "units_sold")
    target: Optional[str] = None
    for col in columns:
        col_lower  = col.lower()
        col_spaced = col_lower.replace("_", " ")
        if col_lower in q or col_spaced in q:
            target = col
            break

    # For aggregate without explicit column, default to first numeric
    if op in ("sum", "avg", "min", "max") and not target and num_cols:
        target = num_cols[0]

    # ── Build SQL for simple aggregate patterns
    sql = ""
    if op == "count":
        sql = "SELECT COUNT(*) AS total_count FROM uploaded_data;"
    elif op in ("sum", "avg", "min", "max") and target:
        fn = op.upper()
        alias = _alias(f"{op}_{target}")
        sql = f"SELECT {fn}({_quote(target)}) AS {alias} FROM uploaded_data;"

    # ── Detect chart hint from question keywords
    hint = ""
    if any(x in q for x in ["distribution", "histogram", "spread", "frequency"]):
        hint = "histogram"
    elif any(x in q for x in ["trend", "over time", "by month", "by year", "time series", "timeline"]):
        hint = "line"
    elif any(x in q for x in ["compare", "by category", "breakdown", "group by", "per ", "each ", "by "]):
        hint = "bar"
    elif any(x in q for x in ["correlation", "scatter", "relationship", "versus", " vs "]):
        hint = "scatter"
    elif any(x in q for x in ["proportion", "share", "percentage", "pie", "composition"]):
        hint = "pie"

    return {**state, "sql": sql, "chart_hint": hint, "status": "OK"}


# ── Node 2 — LLM SQL generation (fallback) ────────────────────────────────────

def node_generate_sql(state: QAState) -> QAState:
    """
    Called only when node_interpret could not produce SQL.
    Sends a structured prompt to Gemini and extracts the SQL.
    """
    if state.get("sql"):      # Already resolved by heuristic
        return state

    columns, _ = get_schema()

    if _client is None and _legacy_model is None:
        return {
            **state,
            "status": "FAILED",
            "outcome": (
                "GEMINI_API_KEY is not configured. "
                "Please add it to your .env file and restart."
            ),
        }

    schema_str = ", ".join(_quote(c) for c in columns)
    prompt = (
        "You are a SQLite expert. Generate exactly one SELECT statement.\n"
        f"Table: uploaded_data\nColumns: {schema_str}\n"
        "Rules: use double-quoted column names; no markdown fences; return SQL only.\n"
        f"Question: {state['question']}\nSQL:"
    )

    try:
        raw = re.sub(r"```[\s\S]*?```", "", _llm_generate(prompt)).replace("`", "").strip()
        m = re.search(r"(SELECT[\s\S]+)", raw, re.IGNORECASE)
        sql = m.group(1).strip() if m else ""
        if sql and not sql.endswith(";"):
            sql += ";"
        return {**state, "sql": sql, "status": "OK"}
    except Exception as exc:
        return {
            **state, "sql": "", "status": "FAILED",
            "outcome": f"SQL generation failed: {exc}",
        }


# ── Node 3 — SQL execution ─────────────────────────────────────────────────────

def node_execute(state: QAState) -> QAState:
    if state.get("status") == "FAILED" or not state.get("sql"):
        return {**state, "result": pd.DataFrame()}
    try:
        df = run_query(state["sql"])
        return {**state, "result": df, "status": "OK"}
    except Exception as exc:
        return {
            **state,
            "result": pd.DataFrame(),
            "status": "FAILED",
            "outcome": f"Query error: {exc}",
        }


# ── Node 4 — Natural language summary ─────────────────────────────────────────

def node_summarize(state: QAState) -> QAState:
    # Don't overwrite an already-set failure message
    if state.get("outcome") and state.get("status") != "FAILED":
        return state

    df: pd.DataFrame = state.get("result", pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            **state,
            "outcome": state.get("outcome") or "No data matched your query.",
            "status": "OK",
        }

    # Single aggregate value — format it nicely without calling the LLM
    if df.shape == (1, 1):
        col = df.columns[0]
        val = df.iloc[0, 0]
        if isinstance(val, float):
            pretty = f"{val:,.2f}"
        elif isinstance(val, int):
            pretty = f"{val:,}"
        else:
            pretty = str(val)
        label = col.replace("_", " ").title()
        return {**state, "outcome": f"**{label}:** {pretty}", "status": "OK"}

    # Multi-row result — ask Gemini for a plain-English summary
    if _client is not None or _legacy_model is not None:
        try:
            cols = df.columns.tolist()
            sample = df.head(5).to_dict(orient="records")
            prompt = (
                "Summarize these SQL query results in 1-2 concise sentences "
                "for a non-technical business user.\n"
                f"Original question: {state['question']}\n"
                f"Columns: {cols}\nSample rows: {sample}"
            )
            return {**state, "outcome": _llm_generate(prompt).strip(), "status": "OK"}
        except Exception:
            pass

    return {
        **state,
        "outcome": f"Found **{len(df):,}** records across {len(df.columns)} columns.",
        "status": "OK",
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_qa_agent():
    """Compile and return the LangGraph Q&A agent."""
    g = StateGraph(QAState)
    g.add_node("interpret",    node_interpret)
    g.add_node("generate_sql", node_generate_sql)
    g.add_node("execute",      node_execute)
    g.add_node("summarize",    node_summarize)

    g.add_edge(START,          "interpret")
    g.add_edge("interpret",    "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute",      "summarize")
    g.add_edge("summarize",    END)

    return g.compile()
