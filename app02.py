import streamlit as st
import pandas as pd
from collections import defaultdict
import re
from utils_agentic import build_validation_graph, build_sql_qa_graph

# === Page Setup ===
st.set_page_config(page_title="Agentic Data Analyzer", layout="wide")

# === Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "submitted_question" not in st.session_state:
    st.session_state.submitted_question = None

# === App Header ===
st.markdown("""
    <div style="text-align: center;">
        <h1>📊 Agentic Data Analyzer</h1>
        <h4>Data Validation, Autofix, Downloading option, Intelligent Answers</h4>
    </div>
""", unsafe_allow_html=True)

# === Step 1: File Upload ===
st.markdown("### Step 1: Upload your customer dataset (.csv or .xlsx)")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
df_cleaned = None

if uploaded_file:
    file_size = uploaded_file.size
    time_estimate = "< 60 seconds" if file_size < 5_000_000 else "> 60 seconds"
    st.info(f"Upload successful! Validation will take {time_estimate}...")

    # Load the file
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    # === Step 2: Validation ===
    with st.spinner("🔍 Validating and cleaning data using agents..."):
        graph = build_validation_graph()
        state = graph.invoke({"df_raw": df_raw})
        df_cleaned = state["df_validated"]
        st.success("✅ Validation completed!")

    # === Step 3: Show Validation Summary ===
    st.markdown("### Step 2: Validation Summary")
    st.markdown(f"**Dataset Dimensions:** {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")

    issue_summary = df_cleaned["validation_issues"]
    issue_tracker = defaultdict(lambda: defaultdict(int))

    for val in issue_summary:
        if val:
            for issue in val.split(";"):
                issue = issue.strip()
                if not issue:
                    continue

                issue_lower = issue.lower()
                if "customer_id" in issue_lower:
                    col = "customer_id"
                elif "name" in issue_lower:
                    col = "name"
                elif "surname" in issue_lower:
                    col = "surname"
                elif "gender" in issue_lower:
                    col = "gender"
                elif "age" in issue_lower:
                    col = "age"
                elif "region" in issue_lower:
                    col = "region"
                elif "job_classification" in issue_lower:
                    col = "job_classification"
                elif "date" in issue_lower:
                    col = "date_joined"
                elif "balance" in issue_lower:
                    col = "balance"
                else:
                    col = "unknown"

                issue_type = issue.split()[0].capitalize()
                issue_tracker[col][issue_type] += 1

    if issue_tracker:
        st.markdown("**Data discrepancies found in the following columns:**")
        for col, issue_types in issue_tracker.items():
            details = ", ".join(f"{v} {k.lower()}" for k, v in issue_types.items())
            st.markdown(f"- **{col}** → {details}")
    else:
        st.markdown("🎉 No validation issues found!")

    # === Step 4: Download Button ===
    csv_clean = df_cleaned.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Cleaned Data", csv_clean, file_name="cleaned_data.csv", mime="text/csv")

    # === Step 5: Q&A Section ===
    st.markdown("---")
    st.markdown("### Step 3: Ask Questions About Your Data")

    # ✅ Don't bind to session state — use local variable only
    question_input = st.text_input("🔎 Type your question", placeholder="e.g., What is the average balance by region?")

    if st.button("Submit Question") and question_input:
        # Store question temporarily
        st.session_state.submitted_question = question_input

    # Process the submitted question
    if st.session_state.submitted_question:
        question = st.session_state.submitted_question
        sql, df_answer, status = "", pd.DataFrame(), "FAILED"

        with st.spinner("💡 Generating answer..."):
            try:
                qa_graph = build_sql_qa_graph()
                result = qa_graph.invoke({"question": question})
                sql = result.get("sql", "")
                df_answer = result.get("result", pd.DataFrame())
                status = result.get("status", "FAILED")
            except Exception as e:
                st.error(f"❌ Gemini failed: {e}")

        st.session_state.chat_history.append((question, sql, df_answer, status))
        st.session_state.submitted_question = None  # Clear after use

# === Step 6: Chat History Display ===
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for q, sql, df_ans, status in st.session_state.chat_history[::-1]:
        st.markdown("#### 🟢 You:")
        st.markdown(f"`{q}`")

        st.markdown("#### 🤖 SQL Generated:")
        st.code(sql if sql else "No SQL generated.")

        st.markdown("#### 📊 Answer:")
        if isinstance(df_ans, pd.DataFrame) and not df_ans.empty:
            st.dataframe(df_ans, use_container_width=True)
        else:
            st.markdown("No results found.")

    # === Clear Chat Button ===
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history.clear()
        st.success("Chat history cleared!")
