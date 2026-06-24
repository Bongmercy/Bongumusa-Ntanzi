"""
FinSight — AI-Powered Financial Statement Analyser
Built with Streamlit + Anthropic Claude + PyMuPDF + Plotly

Run with:
    streamlit run financial_analyser.py
"""

import json
import re
import streamlit as st
import anthropic
import plotly.graph_objects as go
import fitz  # PyMuPDF


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BwAise — Financial Analyser",
    page_icon="📊",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e8eaf0; }

    /* Hide default Streamlit menu & footer */
    #MainMenu, footer { visibility: hidden; }

    /* Metric cards */
    .metric-box {
        background: #1a1d27;
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }
    .metric-label  { font-size: 12px; color: #8a8fa8; margin-bottom: 2px; }
    .metric-value  { font-size: 22px; font-weight: 600; }
    .metric-note   { font-size: 11px; color: #555a72; margin-top: 3px; }

    /* Score card */
    .score-card {
        background: #1a1d27;
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }

    /* Tip items */
    .tip-high   { border-left: 3px solid #f5525a; background: rgba(245,82,90,0.07);  border-radius: 0 8px 8px 0; padding: 12px 14px; margin-bottom: 8px; }
    .tip-medium { border-left: 3px solid #f5a623; background: rgba(245,166,35,0.07); border-radius: 0 8px 8px 0; padding: 12px 14px; margin-bottom: 8px; }
    .tip-low    { border-left: 3px solid #3ecf8e; background: rgba(62,207,142,0.07); border-radius: 0 8px 8px 0; padding: 12px 14px; margin-bottom: 8px; }

    .tip-title  { font-weight: 600; font-size: 14px; margin-bottom: 3px; }
    .tip-detail { font-size: 13px; color: #8a8fa8; line-height: 1.55; }

    .badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
        margin-bottom: 6px;
    }
    .badge-high   { background: rgba(245,82,90,0.18);  color: #f5525a; }
    .badge-medium { background: rgba(245,166,35,0.18); color: #f5a623; }
    .badge-low    { background: rgba(62,207,142,0.18); color: #3ecf8e; }

    /* Section headers */
    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #555a72;
        margin: 1.5rem 0 0.5rem;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background-color: #1a1d27 !important;
        color: #e8eaf0 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────

JSON_SCHEMA = """\
Return ONLY a valid JSON object — no markdown fences, no extra text — with this structure:
{
  "score": 72,
  "health": "Fair",
  "summary": "Two-sentence plain-English summary of financial health.",
  "metrics": [
    {"label": "Savings rate",    "value": "12%",      "note": "below recommended 20%"},
    {"label": "Debt ratio",      "value": "38%",      "note": "moderate pressure"},
    {"label": "Net worth",       "value": "R160,000", "note": "positive"},
    {"label": "Monthly surplus", "value": "R1,800",   "note": "available to deploy"}
  ],
  "tips": [
    {"priority": "High",   "title": "Short actionable title", "detail": "Specific advice in 1-2 sentences."},
    {"priority": "High",   "title": "Another high tip",        "detail": "..."},
    {"priority": "Medium", "title": "Medium priority tip",     "detail": "..."},
    {"priority": "Medium", "title": "Another medium tip",      "detail": "..."},
    {"priority": "Low",    "title": "Low priority tip",        "detail": "..."}
  ],
  "chartLabels":   ["Jan","Feb","Mar","Apr","May","Jun"],
  "chartIncome":   [30000,32000,31000,33000,31500,34000],
  "chartExpenses": [25000,26000,24000,27000,25500,26500]
}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF using PyMuPDF."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text[:8000]  # Limit to avoid token overflow


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


def score_color(score: int) -> str:
    if score >= 75:
        return "#3ecf8e"
    elif score >= 50:
        return "#f5a623"
    return "#f5525a"


def health_emoji(health: str) -> str:
    return {"Good": "🟢", "Fair": "🟡", "Poor": "🔴"}.get(health, "⚪")


def call_claude(api_key: str, messages: list, system: str) -> str:
    """Call the Anthropic API and return text response."""
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=system,
        messages=messages,
    )
    return "".join(block.text for block in response.content if hasattr(block, "text"))


def render_chart(labels: list, income: list, expenses: list):
    """Render a grouped bar chart with Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Income",
        x=labels,
        y=income,
        marker_color="#4f8ef7",
        marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        name="Expenses",
        x=labels,
        y=expenses,
        marker_color="#f5a623",
        marker_line_width=0,
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#1a1d27",
        paper_bgcolor="#0f1117",
        font_color="#8a8fa8",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            tickprefix="R",
            tickformat=",",
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_results(data: dict):
    """Render the full results panel."""
    sc = score_color(data.get("score", 0))

    st.markdown("---")

    # ── Score card ──────────────────────────────────────────────
    st.markdown("##### 📊 Overview")
    col_score, col_summary = st.columns([1, 2])
    with col_score:
        st.markdown(f"""
        <div class="score-card" style="text-align:center;">
            <div style="font-size:11px;color:#8a8fa8;text-transform:uppercase;letter-spacing:.05em;">Health score</div>
            <div style="font-size:52px;font-weight:700;color:{sc};line-height:1.1;">{data.get("score","—")}</div>
            <div style="font-size:13px;color:#8a8fa8;">/100</div>
            <div style="font-size:16px;margin-top:6px;">{health_emoji(data.get("health",""))} {data.get("health","")}</div>
        </div>""", unsafe_allow_html=True)
    with col_summary:
        st.markdown(f"""
        <div class="score-card" style="height:100%;display:flex;align-items:center;">
            <p style="color:#8a8fa8;font-size:15px;line-height:1.7;margin:0;">{data.get("summary","")}</p>
        </div>""", unsafe_allow_html=True)

    # ── Key metrics ─────────────────────────────────────────────
    st.markdown("##### 📌 Key metrics")
    metrics = data.get("metrics", [])
    cols = st.columns(len(metrics) if metrics else 1)
    for col, m in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{m["label"]}</div>
                <div class="metric-value">{m["value"]}</div>
                <div class="metric-note">{m["note"]}</div>
            </div>""", unsafe_allow_html=True)

    # ── Chart ────────────────────────────────────────────────────
    st.markdown("##### 📈 Income vs expenses")
    render_chart(
        data.get("chartLabels", []),
        data.get("chartIncome", []),
        data.get("chartExpenses", []),
    )

    # ── Tips ─────────────────────────────────────────────────────
    st.markdown("##### 💡 Improvement tips")
    tip_class = {"High": "tip-high", "Medium": "tip-medium", "Low": "tip-low"}
    badge_class = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}

    for tip in data.get("tips", []):
        p = tip.get("priority", "Low")
        st.markdown(f"""
        <div class="{tip_class.get(p, 'tip-low')}">
            <span class="badge {badge_class.get(p, 'badge-low')}">{p}</span>
            <div class="tip-title">{tip["title"]}</div>
            <div class="tip-detail">{tip["detail"]}</div>
        </div>""", unsafe_allow_html=True)


# ── App layout ─────────────────────────────────────────────────────────────────

st.markdown("# 📊 BwAise")
st.markdown("**AI-powered financial statement analyser** — upload a PDF or enter your figures for an instant review and personalised tips.")
st.markdown("---")

# API Key input
with st.expander("🔑 Anthropic API key (required)", expanded=True):
    api_key = st.text_input(
        "Paste your API key",
        type="password",
        placeholder="sk-ant-api03-…",
        help="Get your key at console.anthropic.com — it is never stored anywhere.",
    )
    st.caption("Your key stays in memory for this session only. [Get a key →](https://console.anthropic.com)")

if not api_key:
    st.info("Enter your Anthropic API key above to get started.")
    st.stop()

st.markdown("---")

# ── Tab selector ──────────────────────────────────────────────────────────────

tab_pdf, tab_manual = st.tabs(["📄  PDF / file upload", "✏️  Manual entry"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PDF upload
# ══════════════════════════════════════════════════════════════════════════════

with tab_pdf:
    st.markdown("Upload a financial statement (PDF, TXT, or CSV).")
    uploaded = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "csv"],
        label_visibility="collapsed",
    )
    stmt_type = st.text_input(
        "What type of statement is this? (optional)",
        placeholder="e.g. personal budget, business P&L, bank statement, annual report…",
    )

    if uploaded and st.button("🔍 Analyse statement", use_container_width=True, key="pdf_btn"):
        with st.spinner("Reading your document and analysing…"):
            try:
                # Extract text
                if uploaded.type == "application/pdf":
                    text = extract_pdf_text(uploaded)
                else:
                    text = uploaded.read().decode("utf-8", errors="ignore")[:8000]

                label = stmt_type or "financial statement"
                prompt = f"Here is a {label}:\n\n{text}\n\n{JSON_SCHEMA}"

                raw = call_claude(
                    api_key,
                    [{"role": "user", "content": prompt}],
                    system="You are an expert financial analyst. Respond ONLY with valid JSON.",
                )
                result = parse_json_response(raw)
                st.session_state["pdf_result"] = result
                st.session_state["pdf_context"] = f"Financial review of a {label}"

            except anthropic.AuthenticationError:
                st.error("❌ Invalid API key. Check your key at console.anthropic.com.")
            except json.JSONDecodeError:
                st.error("❌ Could not parse the AI response. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if "pdf_result" in st.session_state:
        render_results(st.session_state["pdf_result"])

        # Follow-up
        st.markdown("---")
        st.markdown("##### 💬 Ask a follow-up question")
        fq = st.text_input("Your question", placeholder="e.g. How can I build an emergency fund faster?", key="pdf_fq")
        if st.button("Ask →", key="pdf_ask"):
            with st.spinner("Thinking…"):
                try:
                    answer = call_claude(
                        api_key,
                        [{"role": "user", "content": f"Context: {st.session_state['pdf_context']}\n\nQuestion: {fq}"}],
                        system="You are a friendly, expert financial advisor. Keep answers concise — 3 to 5 sentences. No jargon.",
                    )
                    st.info(answer)
                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Manual entry
# ══════════════════════════════════════════════════════════════════════════════

with tab_manual:

    st.markdown("##### Income (monthly)")
    col1, col2 = st.columns(2)
    with col1:
        income_main  = st.number_input("Main income / salary (R)", min_value=0, step=500, value=0)
    with col2:
        income_other = st.number_input("Other income (R)", min_value=0, step=100, value=0)

    st.markdown("##### Expenses (monthly)")
    c1, c2 = st.columns(2)
    with c1:
        rent          = st.number_input("Housing / rent / bond (R)",  min_value=0, step=100, value=0)
        transport     = st.number_input("Transport (R)",               min_value=0, step=100, value=0)
        debt          = st.number_input("Debt repayments (R)",         min_value=0, step=100, value=0)
        savings       = st.number_input("Savings / investments (R)",   min_value=0, step=100, value=0)
    with c2:
        food          = st.number_input("Food & groceries (R)",        min_value=0, step=100, value=0)
        utilities     = st.number_input("Utilities (R)",               min_value=0, step=50,  value=0)
        entertainment = st.number_input("Entertainment / dining (R)",  min_value=0, step=100, value=0)
        other_exp     = st.number_input("Other expenses (R)",          min_value=0, step=50,  value=0)

    st.markdown("##### Balance sheet (totals, not monthly)")
    b1, b2 = st.columns(2)
    with b1:
        assets      = st.number_input("Total assets (R)",      min_value=0, step=1000, value=0)
    with b2:
        liabilities = st.number_input("Total liabilities (R)", min_value=0, step=1000, value=0)

    context = st.text_input(
        "Context (optional — personalises the advice)",
        placeholder="e.g. freelance developer in Durban, saving for a house, 2 dependants…",
    )

    total_income   = income_main + income_other
    total_expenses = rent + food + transport + utilities + debt + entertainment + savings + other_exp
    surplus        = total_income - total_expenses
    net_worth      = assets - liabilities

    # Live summary
    if total_income > 0 or total_expenses > 0:
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Total income",   f"R{total_income:,.0f}")
        col_s2.metric("Total expenses", f"R{total_expenses:,.0f}")
        col_s3.metric("Surplus / deficit", f"R{surplus:,.0f}", delta_color="normal" if surplus >= 0 else "inverse")

    if st.button("🔍 Analyse my finances", use_container_width=True, key="manual_btn"):
        if total_income == 0 and total_expenses == 0:
            st.warning("Please enter at least some income or expense figures.")
        else:
            with st.spinner("Crunching your numbers…"):
                try:
                    data_prompt = f"""\
Monthly financial data (ZAR):
- Main income: R{income_main:,.0f}, other income: R{income_other:,.0f}, total income: R{total_income:,.0f}
- Housing/rent: R{rent:,.0f}, food: R{food:,.0f}, transport: R{transport:,.0f}
- Utilities: R{utilities:,.0f}, debt repayments: R{debt:,.0f}, entertainment: R{entertainment:,.0f}
- Savings/investments: R{savings:,.0f}, other: R{other_exp:,.0f}
- Total expenses: R{total_expenses:,.0f}
- Monthly surplus/deficit: R{surplus:,.0f}
- Total assets: R{assets:,.0f}, total liabilities: R{liabilities:,.0f}, net worth: R{net_worth:,.0f}
- Context: {context or "not provided"}

{JSON_SCHEMA}"""

                    raw = call_claude(
                        api_key,
                        [{"role": "user", "content": data_prompt}],
                        system="You are an expert financial analyst. Respond ONLY with valid JSON.",
                    )
                    result = parse_json_response(raw)
                    st.session_state["manual_result"] = result
                    st.session_state["manual_context"] = (
                        f"Income R{total_income:,.0f}/month, expenses R{total_expenses:,.0f}/month, "
                        f"net worth R{net_worth:,.0f}. {context}"
                    )

                except anthropic.AuthenticationError:
                    st.error("❌ Invalid API key. Check your key at console.anthropic.com.")
                except json.JSONDecodeError:
                    st.error("❌ Could not parse the AI response. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if "manual_result" in st.session_state:
        render_results(st.session_state["manual_result"])

        # Follow-up
        st.markdown("---")
        st.markdown("##### 💬 Ask a follow-up question")
        fq2 = st.text_input("Your question", placeholder="e.g. What's the fastest way to improve my score?", key="manual_fq")
        if st.button("Ask →", key="manual_ask"):
            with st.spinner("Thinking…"):
                try:
                    answer = call_claude(
                        api_key,
                        [{"role": "user", "content": f"Context: {st.session_state['manual_context']}\n\nQuestion: {fq2}"}],
                        system="You are a friendly, expert financial advisor. Keep answers concise — 3 to 5 sentences. No jargon.",
                    )
                    st.info(answer)
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("FinSight · Powered by Claude (Anthropic) · For informational purposes only — not financial advice.")