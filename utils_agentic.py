# ✅ Agentic Utilities for Structured Data QA using LangGraph + Gemini
import os
import re
import sqlite3
import pandas as pd
from typing import TypedDict, Literal
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

# 🔐 Load Gemini API key from .env
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# === Shared State Definition for LangGraph ===
class GraphState(TypedDict):
    df_raw: pd.DataFrame
    df_validated: pd.DataFrame
    question: str
    sql: str
    result: pd.DataFrame
    status: Literal["OK", "FAILED"]

# === 🧹 Agent 1: Data Validation and Cleaning Agent ===
def validate_data(state: GraphState) -> GraphState:
    df = state["df_raw"].copy()
    issues, fixes = [], []

    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    expected_cols = ['customer_id', 'name', 'surname', 'gender', 'age', 'region', 'job_classification', 'date_joined', 'balance']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    balance_thresh = df['balance'].quantile(0.99) if pd.api.types.is_numeric_dtype(df['balance']) else None
    existing_ids = set(df['customer_id'].dropna().astype(str))
    auto_id_counter = 100001

    for i, row in df.iterrows():
        row_issues, row_fixes = [], []

        if pd.isna(row['customer_id']):
            new_id = f"AUTO-{auto_id_counter}"
            while new_id in existing_ids:
                auto_id_counter += 1
                new_id = f"AUTO-{auto_id_counter}"
            df.at[i, 'customer_id'] = new_id
            existing_ids.add(new_id)
            auto_id_counter += 1
            row_issues.append("Missing customer_id")
            row_fixes.append(f"Auto-generated ID: {new_id}")

        if pd.isna(row['name']):
            df.at[i, 'name'] = "Unknown"
            row_issues.append("Missing name")
            row_fixes.append("Filled name with 'Unknown'")

        if pd.isna(row['surname']):
            df.at[i, 'surname'] = "Unknown"
            row_issues.append("Missing surname")
            row_fixes.append("Filled surname with 'Unknown'")

        if pd.isna(row['gender']):
            df.at[i, 'gender'] = "Not Specified"
            row_issues.append("Missing gender")
            row_fixes.append("Filled gender with 'Not Specified'")

        if pd.isna(row['region']):
            df.at[i, 'region'] = "Unknown"
            row_issues.append("Missing region")
            row_fixes.append("Filled region with 'Unknown'")

        if pd.isna(row['job_classification']):
            df.at[i, 'job_classification'] = "Other"
            row_issues.append("Missing job_classification")
            row_fixes.append("Filled job_classification with 'Other'")

        if pd.isna(row['age']):
            avg_age = df['age'].mean()
            df.at[i, 'age'] = avg_age
            row_issues.append("Missing age")
            row_fixes.append(f"Filled age with average age {avg_age:.1f}")

        if pd.isna(row['balance']) or row['balance'] < 0:
            df.at[i, 'balance'] = 0.0
            row_issues.append("Missing or negative balance")
            row_fixes.append("Filled balance with 0")

        try:
            pd.to_datetime(row['date_joined'], format="%d.%b.%y")
        except:
            df.at[i, 'date_joined'] = "01.Jan.00"
            row_issues.append("Bad date format")
            row_fixes.append("Filled with default date 01.Jan.00")

        if balance_thresh and row['balance'] > balance_thresh:
            row_issues.append("Outlier in balance")

        issues.append("; ".join(row_issues))
        fixes.append("; ".join(row_fixes))

    df['validation_issues'] = issues
    df['fix_suggestions'] = fixes

    return {"df_raw": df, "df_validated": df, "status": "OK"}

# === 💾 Agent 2: Save Cleaned Data to SQLite ===
def store_to_sqlite(state: GraphState) -> GraphState:
    df = state["df_validated"]
    conn = sqlite3.connect("bank_customer.db")
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    return state

# === 🤖 Agent 3: SQL Generation Agent using Gemini ===
def run_llm_sql(state: GraphState) -> GraphState:
    try:
        conn = sqlite3.connect("bank_customer.db")
        columns = pd.read_sql("SELECT * FROM customers LIMIT 1", conn).columns.tolist()
        conn.close()

        prompt = f"""
Given a table named 'customers' with columns: {', '.join(columns)}.

Note: "age_group" is not a column but can be derived using:
CASE 
  WHEN age < 30 THEN 'Young'
  WHEN age < 60 THEN 'Middle-aged'
  ELSE 'Senior'
END AS age_group.

Convert the following natural language question into a valid SQLite SQL query using this logic if needed.
Only return the SQL.
Question: {state['question']}
SQL:
"""
        response = gemini_model.generate_content(prompt)
        generated_text = response.text.strip()

        sql_match = re.search(r"(SELECT[\s\S]+?);", generated_text, re.IGNORECASE)
        sql = sql_match.group(1) + ";" if sql_match else ""

        return {**state, "sql": sql}
    except Exception as e:
        print("Gemini Error:", e)
        return {**state, "sql": "", "status": "FAILED"}

# === 🧪 Agent 4: Execute SQL Query ===
def execute_sql(state: GraphState) -> GraphState:
    try:
        conn = sqlite3.connect("bank_customer.db")
        df_result = pd.read_sql(state["sql"], conn)
        conn.close()
        return {**state, "result": df_result, "status": "OK"}
    except Exception:
        return {**state, "result": pd.DataFrame(), "status": "FAILED"}

# === ⚙️ LangGraph Builder: Validation Graph ===
def build_validation_graph():
    graph = StateGraph(GraphState)
    graph.add_node("validate", RunnableLambda(validate_data))
    graph.add_node("store", RunnableLambda(store_to_sqlite))
    graph.set_entry_point("validate")
    graph.add_edge("validate", "store")
    graph.set_finish_point("store")
    return graph.compile()

# === ⚙️ LangGraph Builder: SQL Q&A Graph ===
def build_sql_qa_graph():
    graph = StateGraph(GraphState)
    graph.add_node("generate_sql", RunnableLambda(run_llm_sql))
    graph.add_node("run_sql", RunnableLambda(execute_sql))
    graph.set_entry_point("generate_sql")
    graph.add_edge("generate_sql", "run_sql")
    graph.set_finish_point("run_sql")
    return graph.compile()
