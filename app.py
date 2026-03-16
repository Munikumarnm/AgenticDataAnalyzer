"""
app.py — DataCopilot: Agentic Data Analyzer
============================================
Layout
  [Header — full width]
  [Left 63%: tabs Upload & Clean | Dashboard] | [Right 37%: Chat Copilot — always visible]

Run with:
  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.agent import build_qa_agent
from backend.data_processor import process_upload
from backend.database import save_dataframe

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataCopilot — Agentic Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  /* Layout & typography */
  .block-container { padding-top: 1.1rem; padding-bottom: 0.5rem; }
  html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }
  #MainMenu, footer, header { visibility: hidden; }

  /* ── App header ── */
  .app-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b3a6b 50%, #0d3b6e 100%);
    border-radius: 14px;
    padding: 18px 28px;
    margin-bottom: 20px;
  }
  .app-header h1 {
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: -0.3px;
  }
  .app-header p {
    color: #94a3b8;
    font-size: 0.84rem;
    margin: 0;
  }

  /* ── KPI cards ── */
  .kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    height: 88px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
  }
  .kpi-label {
    font-size: 0.66rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 5px;
  }

  /* ── Section headers ── */
  .section-hdr {
    font-size: 0.75rem;
    font-weight: 700;
    color: #334155;
    border-left: 3px solid #3b82f6;
    padding-left: 8px;
    margin: 18px 0 10px 0;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  /* ── Copilot panel header ── */
  .copilot-header {
    background: #0f172a;
    border-radius: 12px 12px 0 0;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0;
  }
  .copilot-status {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
    box-shadow: 0 0 6px #22c55e88;
  }
  .copilot-title {
    color: #ffffff;
    font-weight: 700;
    font-size: 0.92rem;
  }
  .copilot-model {
    color: #475569;
    font-size: 0.75rem;
    margin-left: auto;
  }

  /* ── Empty-state placeholder ── */
  .empty-state {
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    color: #64748b;
    margin-top: 12px;
  }
  .empty-state .icon { font-size: 2.2rem; margin-bottom: 8px; }
  .empty-state .title { font-size: 1rem; font-weight: 600; margin: 0 0 4px; }
  .empty-state .sub { font-size: 0.82rem; margin: 0; }

  /* ── Streamlit tab styling ── */
  .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.85rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session-state defaults ────────────────────────────────────────────────────
_defaults = {
    "data_ready":    False,
    "clean_df":      pd.DataFrame(),
    "report":        {},
    "chat_history":  [],
    "qa_agent":      None,
    "pending_q":     "",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-header">
  <h1>📊 DataCopilot — Agentic Data Analyzer</h1>
  <p>Upload CSV / Excel → auto-clean & anomaly removal → interactive dashboard → ask questions in plain English</p>
</div>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kpi(col_widget, value: str, label: str) -> None:
    col_widget.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _section(title: str) -> None:
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)


def _render_charts_2col(figures: list) -> None:
    """Render a list of Plotly figures in a 2-column grid."""
    for i in range(0, len(figures), 2):
        pair = figures[i : i + 2]
        cols = st.columns(len(pair))
        for j, fig in enumerate(pair):
            cols[j].plotly_chart(fig, use_container_width=True)


def _histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(
        df, x=col, nbins=35,
        title=col.replace("_", " ").title(),
        color_discrete_sequence=["#3b82f6"],
        template="plotly_white",
    )
    fig.update_layout(
        showlegend=False, height=240,
        margin=dict(l=8, r=8, t=36, b=8),
        title_font_size=12,
    )
    return fig


def _boxplot(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.box(
        df, y=col,
        title=col.replace("_", " ").title(),
        color_discrete_sequence=["#f59e0b"],
        template="plotly_white",
        points="outliers",
    )
    fig.update_layout(
        showlegend=False, height=240,
        margin=dict(l=8, r=8, t=36, b=8),
        title_font_size=12,
    )
    return fig


def _category_chart(df: pd.DataFrame, col: str) -> go.Figure:
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    if df[col].nunique() <= 7:
        fig = px.pie(
            vc, names=col, values="count",
            title=col.replace("_", " ").title(),
            color_discrete_sequence=px.colors.qualitative.Plotly,
            template="plotly_white",
        )
        fig.update_traces(textinfo="percent+label")
    else:
        fig = px.bar(
            vc.head(15), x=col, y="count",
            title=col.replace("_", " ").title(),
            color_discrete_sequence=["#8b5cf6"],
            template="plotly_white",
        )
    fig.update_layout(
        showlegend=False, height=240,
        margin=dict(l=8, r=8, t=36, b=8),
        title_font_size=12,
    )
    return fig


def _process_question(question: str) -> None:
    """Run the Q&A agent for `question` and append result to chat_history."""
    st.session_state.chat_history.append({"role": "user", "content": question})
    try:
        if st.session_state.qa_agent is None:
            st.session_state.qa_agent = build_qa_agent()

        out = st.session_state.qa_agent.invoke({"question": question})
        
        # Debug: check what we got back
        outcome_text = out.get("outcome", "") or out.get("content", "")
        if not outcome_text:
            outcome_text = "I could not find an answer."
        
        st.session_state.chat_history.append(
            {
                "role":       "assistant",
                "content":    outcome_text,
                "sql":        out.get("sql") or "",
                "result":     out.get("result") if isinstance(out.get("result"), pd.DataFrame) else pd.DataFrame(),
                "chart_hint": out.get("chart_hint") or "",
            }
        )
    except Exception as exc:
        st.session_state.chat_history.append(
            {
                "role":       "assistant",
                "content":    f"Error: {exc}",
                "sql":        "",
                "result":     pd.DataFrame(),
                "chart_hint": "",
            }
        )


# ── Process any pending question BEFORE rendering the layout ──────────────────
# This ensures chat_history is populated before the right panel renders,
# so messages appear immediately without needing a second rerun.
if st.session_state.get("pending_q") and st.session_state.data_ready:
    _q = st.session_state.pending_q
    st.session_state.pending_q = ""
    with st.spinner("Thinking…"):
        _process_question(_q)
    st.rerun()

# ── Main two-column layout ─────────────────────────────────────────────────────
left, right = st.columns([1.75, 1], gap="medium")

# ══════════════════════════════════════════════════════════════════
# LEFT — Upload & Dashboard tabs
# ══════════════════════════════════════════════════════════════════
with left:
    upload_tab, dashboard_tab = st.tabs(["📁  Upload & Clean", "📈  Dashboard"])

    # ── Upload & Clean ────────────────────────────────────────────
    with upload_tab:
        uploaded = st.file_uploader(
            "Drag & drop your file here, or click to browse",
            type=["csv", "xlsx", "xls"],
        )

        remove_anom = st.toggle(
            "Remove anomalies (IQR outlier detection)",
            value=False,
            help=(
                "Uses Tukey's IQR × 1.5 fence to identify rows that are "
                "statistical outliers in at least one numeric column and removes them."
            ),
        )

        if uploaded:
            with st.spinner("Processing your data…"):
                try:
                    raw_df = (
                        pd.read_csv(uploaded)
                        if uploaded.name.lower().endswith(".csv")
                        else pd.read_excel(uploaded)
                    )
                    clean_df, report = process_upload(raw_df, remove_anomalies=remove_anom)
                    save_dataframe(clean_df)
                    st.session_state.clean_df    = clean_df
                    st.session_state.report      = report
                    st.session_state.data_ready  = True
                    st.session_state.chat_history = []   # reset chat on new upload
                    st.session_state.qa_agent     = None # rebuild agent for new schema
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")
                    st.session_state.data_ready = False

        if st.session_state.data_ready:
            df  = st.session_state.clean_df
            rpt = st.session_state.report

            # Processing summary KPIs
            _section("Processing Summary")
            k1, k2, k3, k4 = st.columns(4)
            _kpi(k1, f"{rpt['rows_cleaned']:,}",           "Rows Cleaned")
            _kpi(k2, str(len(df.columns)),                 "Columns")
            _kpi(k3, f"{sum(rpt['missing_filled'].values()):,}", "Cells Fixed")
            _kpi(k4, f"{rpt['anomalies_removed']:,}",      "Anomalies Removed")

            # Missing-values detail (if any)
            if rpt["missing_filled"]:
                with st.expander("Missing value fills", expanded=False):
                    mv_df = pd.DataFrame(
                        [{"Column": c, "Cells Filled": n}
                         for c, n in rpt["missing_filled"].items()]
                    )
                    st.dataframe(mv_df, use_container_width=True, hide_index=True)

            # Download button
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Download Cleaned Data (CSV)",
                data=csv_bytes,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )

            # Data preview
            _section("Data Preview (first 20 rows)")
            st.dataframe(df.head(20), use_container_width=True, height=260)

            # Column statistics
            _section("Column Statistics")
            summary_rows = []
            for c in df.columns:
                row: dict = {
                    "Column":  c,
                    "Type":    str(df[c].dtype),
                    "Missing": int(df[c].isna().sum()),
                    "Unique":  int(df[c].nunique()),
                    "Min":     None,
                    "Max":     None,
                    "Mean":    None,
                }
                if pd.api.types.is_numeric_dtype(df[c]):
                    row["Min"]  = round(float(df[c].min()),  2)
                    row["Max"]  = round(float(df[c].max()),  2)
                    row["Mean"] = round(float(df[c].mean()), 2)
                summary_rows.append(row)
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.markdown(
                """
<div class="empty-state">
  <div class="icon">📂</div>
  <p class="title">No data uploaded yet</p>
  <p class="sub">Supports CSV, XLSX, and XLS files</p>
</div>
""",
                unsafe_allow_html=True,
            )

    # ── Dashboard ─────────────────────────────────────────────────
    with dashboard_tab:
        if not st.session_state.data_ready:
            st.info("Upload a dataset in the **Upload & Clean** tab to see your dashboard.")
        else:
            df        = st.session_state.clean_df
            num_cols  = df.select_dtypes(include="number").columns.tolist()
            cat_cols  = [
                c for c in df.select_dtypes(include="object").columns
                if 2 <= df[c].nunique() <= 30
            ]
            date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

            # ── Dataset overview KPIs
            _section("Dataset Overview")
            o1, o2, o3, o4 = st.columns(4)
            _kpi(o1, f"{len(df):,}",     "Total Records")
            _kpi(o2, str(len(num_cols)), "Numeric Columns")
            _kpi(o3, str(len(cat_cols)), "Category Columns")
            _kpi(o4, str(len(date_cols)),"Date Columns")

            # ── Numeric distributions
            if num_cols:
                _section("Distributions")
                hist_figs = [_histogram(df, c) for c in num_cols[:6]]
                _render_charts_2col(hist_figs)

            # ── Box plots (outlier view)
            if num_cols:
                _section("Box Plots — Outlier View")
                box_figs = [_boxplot(df, c) for c in num_cols[:4]]
                _render_charts_2col(box_figs)

            # ── Category breakdowns
            if cat_cols:
                _section("Category Breakdowns")
                cat_figs = [_category_chart(df, c) for c in cat_cols[:6]]
                _render_charts_2col(cat_figs)

            # ── Time series (if date + numeric columns exist)
            if date_cols and num_cols:
                _section("Time Series")
                ts_col_left, ts_col_right = st.columns([3, 1])
                with ts_col_right:
                    ts_metric = st.selectbox(
                        "Metric", num_cols, key="ts_metric_sel", label_visibility="collapsed"
                    )
                ts_df = (
                    df[[date_cols[0], ts_metric]].dropna().sort_values(date_cols[0])
                )
                fig_ts = px.line(
                    ts_df, x=date_cols[0], y=ts_metric,
                    title=f"{ts_metric.replace('_', ' ').title()} Over Time",
                    color_discrete_sequence=["#10b981"],
                    template="plotly_white",
                )
                fig_ts.update_layout(height=280, margin=dict(l=8, r=8, t=40, b=8))
                st.plotly_chart(fig_ts, use_container_width=True)

            # ── Correlation heatmap (needs 3+ numeric cols)
            if len(num_cols) >= 3:
                _section("Correlation Matrix")
                corr = df[num_cols[:10]].corr()
                fig_corr = go.Figure(
                    go.Heatmap(
                        z=corr.values,
                        x=corr.columns.tolist(),
                        y=corr.columns.tolist(),
                        colorscale="RdBu",
                        zmid=0,
                        text=np.round(corr.values, 2),
                        texttemplate="%{text}",
                        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
                    )
                )
                fig_corr.update_layout(
                    height=420,
                    margin=dict(l=8, r=8, t=16, b=8),
                    template="plotly_white",
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # ── Scatter matrix (2–5 numeric cols)
            if 2 <= len(num_cols) <= 5:
                _section("Scatter Matrix")
                fig_scatter = px.scatter_matrix(
                    df,
                    dimensions=num_cols[:5],
                    color=cat_cols[0] if cat_cols else None,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    template="plotly_white",
                )
                fig_scatter.update_traces(
                    diagonal_visible=False,
                    marker=dict(size=3, opacity=0.5),
                )
                fig_scatter.update_layout(height=480, margin=dict(l=8, r=8, t=16, b=8))
                st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# RIGHT — Chat Copilot (always visible alongside the main content)
# ══════════════════════════════════════════════════════════════════
with right:

    # ── Panel header
    status_dot_color = "#22c55e" if st.session_state.data_ready else "#94a3b8"
    st.markdown(
        f"""
<div class="copilot-header">
  <span class="copilot-status" style="background:{status_dot_color};
        box-shadow:0 0 6px {status_dot_color}88;"></span>
  <span class="copilot-title">Data Copilot</span>
  <span class="copilot-model">Gemini 2.5 Flash</span>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Message area — no fixed height so latest answer is always visible
    msg_area = st.container()
    with msg_area:
        if not st.session_state.data_ready:
            st.markdown(
                """
<div style="text-align:center;color:#94a3b8;padding:50px 16px;">
  <div style="font-size:2.4rem;">💬</div>
  <p style="font-weight:600;color:#64748b;margin:8px 0 4px;">
    Upload data to start chatting
  </p>
  <p style="font-size:0.81rem;margin:0;">
    Ask questions in plain English — I'll convert them to SQL and explain the results.
  </p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            if not st.session_state.chat_history:
                # Welcome message on first load
                df_p    = st.session_state.clean_df
                nc_p    = df_p.select_dtypes(include="number").columns.tolist()
                cc_p    = [
                    c for c in df_p.select_dtypes(include="object").columns
                    if df_p[c].nunique() <= 20
                ]
                with st.chat_message("assistant", avatar="📊"):
                    st.write(
                        f"Hi! Your dataset has **{len(df_p):,} rows** and "
                        f"**{len(df_p.columns)} columns**."
                    )
                    if nc_p:
                        st.write(f"Numeric: `{'`, `'.join(nc_p[:4])}`")
                    if cc_p:
                        st.write(f"Categories: `{'`, `'.join(cc_p[:3])}`")
                    st.caption("Try a suggested question below, or type your own.")
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        with st.chat_message("user"):
                            st.write(msg["content"])
                    else:
                        with st.chat_message("assistant", avatar="📊"):
                            st.markdown(msg["content"])

                            # SQL query — always visible so users can learn from it
                            if msg.get("sql"):
                                st.caption("SQL used:")
                                st.code(msg["sql"], language="sql")

                            # Result table
                            result = msg.get("result")
                            if isinstance(result, pd.DataFrame) and not result.empty:
                                st.dataframe(
                                    result, use_container_width=True, height=130
                                )
                                # Auto-generate a small chart when result has both
                                # categorical and numeric columns and more than 1 row
                                r_num = result.select_dtypes(include="number").columns.tolist()
                                r_cat = result.select_dtypes(include="object").columns.tolist()
                                if r_cat and r_num and len(result) > 1:
                                    mini_fig = px.bar(
                                        result.head(20),
                                        x=r_cat[0],
                                        y=r_num[0],
                                        color_discrete_sequence=["#3b82f6"],
                                        template="plotly_white",
                                    )
                                    mini_fig.update_layout(
                                        height=185,
                                        showlegend=False,
                                        margin=dict(l=4, r=4, t=10, b=4),
                                        xaxis_title="",
                                        yaxis_title="",
                                    )
                                    st.plotly_chart(mini_fig, use_container_width=True)

    # ── Suggested questions
    if st.session_state.data_ready:
        df_s  = st.session_state.clean_df
        nc_s  = df_s.select_dtypes(include="number").columns.tolist()
        cc_s  = [
            c for c in df_s.select_dtypes(include="object").columns
            if df_s[c].nunique() <= 20
        ]
        suggestions = ["How many rows are in the dataset?"]
        if nc_s:
            suggestions.append(f"What is the average {nc_s[0].replace('_', ' ')}?")
            suggestions.append(f"What is the maximum {nc_s[0].replace('_', ' ')}?")
        if cc_s:
            suggestions.append(
                f"Show the count breakdown by {cc_s[0].replace('_', ' ')}"
            )

        with st.expander("💡 Suggested questions", expanded=True):
            sc1, sc2 = st.columns(2)
            cols_cycle = [sc1, sc2, sc1, sc2]
            for idx, sugg in enumerate(suggestions[:4]):
                if cols_cycle[idx].button(sugg, key=f"sugg_{idx}", use_container_width=True):
                    st.session_state.pending_q = sugg
                    st.rerun()

    # ── Clear conversation button
    if st.session_state.chat_history:
        if st.button("🗑  Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Chat input — contained within the right panel
    _placeholder = (
        "Ask about your data…"
        if st.session_state.data_ready
        else "Upload a dataset first to enable chat"
    )
    _question = st.chat_input(_placeholder, disabled=not st.session_state.data_ready)
    if _question and st.session_state.data_ready:
        st.session_state.pending_q = _question
        st.rerun()
