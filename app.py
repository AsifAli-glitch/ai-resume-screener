import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import plotly.graph_objects as go
import plotly.express as px

from src.parser import extract_text_from_pdf, extract_text_from_docx
from src.preprocessor import clean_resume
from src.extractor import extract_skills, extract_contact_info
from src.matcher import weighted_match_score, match_tfidf, match_semantic, score_label

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener & Ranker",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    padding: 2.5rem 2.8rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 0 20px 60px rgba(79,70,229,0.35);
}
.hero h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p { font-size: 1rem; opacity: 0.88; margin: 0; font-weight: 300; }

/* Score badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Stat cards */
.stat-grid { display: flex; gap: 1rem; margin: 1.2rem 0; }
.stat-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
}
.stat-card:hover { transform: translateY(-4px); border-color: rgba(139,92,246,0.45); }
.stat-value { font-size: 1.9rem; font-weight: 700; color: #a78bfa; margin-bottom: 0.15rem; }
.stat-label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; }

/* Skill pills */
.pill-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem; }
.pill {
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
}
.pill-green  { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.pill-red    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.pill-indigo { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.25); }

/* Rank badge */
.rank-1 { color: #fbbf24; font-size: 1.4rem; }
.rank-2 { color: #9ca3af; font-size: 1.4rem; }
.rank-3 { color: #b45309; font-size: 1.4rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 20px; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.92rem; font-weight: 500; color: #9ca3af;
    background: transparent; border-radius: 4px; height: 46px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #818cf8; font-weight: 700; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f5f3ff 0%, #eef2ff 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = None
if 'jd' not in st.session_state:
    st.session_state.jd = ""
if 'shortlist' not in st.session_state:
    st.session_state.shortlist = {}   # filename -> bool
if 'notes' not in st.session_state:
    st.session_state.notes = {}       # filename -> str

# ─────────────────────────────────────────────
# DICTIONARIES & TEMPLATES
# ─────────────────────────────────────────────
JD_TEMPLATES = {
    "Custom (Clear / Paste your own)": "",
    "Senior Python Developer": (
        "We are looking for a Senior Python Developer with 5+ years of experience. "
        "The ideal candidate must be highly proficient in Python, object-oriented programming, "
        "data structures, and building robust backend services using Flask or Django. "
        "Experience with databases like PostgreSQL, MySQL, and containerization using Docker is required. "
        "Strong understanding of Git, software testing, clean architecture, and API design is expected."
    ),
    "Machine Learning Engineer": (
        "Seeking a Machine Learning Engineer to design, build, and deploy production ML models. "
        "Required skills include Python, machine learning algorithms, deep learning (TensorFlow, PyTorch), "
        "data analysis and visualization using Pandas, Numpy, Matplotlib, and Scikit-learn. "
        "Experience with big data tools (Spark), SQL databases, and deploying models on cloud platforms (AWS, GCP) is essential."
    ),
    "Frontend React Developer": (
        "We are looking for a Frontend Web Developer specializing in building interactive and responsive UI. "
        "Must have strong experience in HTML, CSS, JavaScript, TypeScript, and the React framework. "
        "Familiarity with CSS utility frameworks (Tailwind, Bootstrap), state management, and Git/GitHub is required. "
        "Experience with Next.js or visual design tooling is a plus."
    ),
    "Full Stack Software Engineer": (
        "We are looking for a Full Stack Software Engineer to build and maintain end-to-end applications. "
        "Technical stack includes React on the front-end, Node.js and Express on the backend, "
        "SQL databases (MySQL, PostgreSQL), and RESTful API integration. "
        "Knowledge of DevOps tools like Git, Docker, CI/CD pipelines, and cloud services (AWS) is highly desirable."
    )
}

INTERVIEW_QUESTIONS = {
    "Python": "How do you manage memory in Python, and what are the differences between lists and generators?",
    "Django": "Can you explain the Django request-response lifecycle and how middleware works?",
    "Flask": "How does Flask handle application context, and what are the pros/cons of Flask vs Django?",
    "Fastapi": "What makes FastAPI faster than traditional frameworks, and how do you implement dependency injection?",
    "Machine Learning": "Explain the bias-variance tradeoff and how you prevent overfitting in model training.",
    "Deep Learning": "What are the common optimization algorithms in deep learning, and when would you use Adam vs SGD?",
    "Tensorflow": "How do you build a custom training loop in TensorFlow, and what is the function of tf.function?",
    "Pytorch": "Explain autograd in PyTorch and how dynamic computation graphs differ from static ones.",
    "Sql": "What are the differences between joins and subqueries, and how do you optimize a slow database query?",
    "Postgresql": "What are index types in PostgreSQL (e.g. B-Tree, GIN, GiST), and when should you use GIN?",
    "Mysql": "Explain transaction isolation levels in InnoDB and how deadlocks are resolved.",
    "React": "What are React Server Components and how do they differ from client components?",
    "Node.Js": "Explain the Node.js event loop and how it handles asynchronous I/O operations.",
    "Docker": "How do you minimize Docker image sizes using multi-stage builds?",
    "Kubernetes": "What is a Kubernetes Pod, and how does service discovery work inside a cluster?",
    "Git": "What is the difference between git merge and git rebase, and when would you use each?",
    "Github": "How do you configure GitHub Actions for continuous integration and automated deployments?",
    "AWS": "What AWS services would you use to build a highly available, fault-tolerant web application backend?",
    "GCP": "How do Google Cloud IAM roles work, and how do you securely manage service accounts?",
    "Data Science": "Walk me through your project architecture when designing a machine learning pipeline from raw data to deployment.",
    "Data Analysis": "Walk me through your data cleaning and preprocessing pipeline for a noisy dataset.",
    "Pandas": "How do you handle missing values or perform efficient grouping/aggregations on large DataFrames in Pandas?",
    "Numpy": "Explain vectorization in NumPy and why it is faster than standard Python loops.",
    "Scikit-Learn": "How do you construct a machine learning Pipeline in scikit-learn for preprocessing and modeling?",
    "HTML": "What is semantic HTML, and why is it important for accessibility and SEO?",
    "CSS": "Explain the CSS Box Model and how flexbox differs from grid layouts.",
    "Tailwind": "What are the advantages of utility-first CSS frameworks like Tailwind over styled components or traditional CSS?",
    "Communication": "Tell me about a time you had to explain a complex technical concept to a non-technical stakeholder.",
    "Problem Solving": "Describe the most challenging bug you've encountered and the step-by-step process you used to debug it."
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
import html
import re

def highlight_skills(text, matched_skills, extra_skills):
    escaped_text = html.escape(text)
    all_highlights = {}
    
    for s in matched_skills:
        all_highlights[s.lower()] = f"<mark style='background-color:#d1fae5;color:#065f46;border-radius:4px;padding:2px 4px;font-weight:600;'>{html.escape(s)}</mark>"
        
    for s in extra_skills:
        all_highlights[s.lower()] = f"<mark style='background-color:#e0e7ff;color:#3730a3;border-radius:4px;padding:2px 4px;font-weight:600;'>{html.escape(s)}</mark>"
        
    sorted_skills = sorted(list(all_highlights.keys()), key=len, reverse=True)
    placeholders = {}
    
    temp_text = escaped_text
    for i, skill in enumerate(sorted_skills):
        placeholder = f"___SKILL_HL_{i}___"
        placeholders[placeholder] = all_highlights[skill]
        
        escaped_skill = re.escape(skill)
        if '+' in skill or '.' in skill:
            pattern = r'(?i)(?:^|\s|\b)' + escaped_skill + r'(?:\s|\b|$)'
        else:
            pattern = r'(?i)\b' + escaped_skill + r'\b'
            
        def repl(match):
            val = match.group(0)
            start_idx = val.lower().find(skill)
            prefix = val[:start_idx]
            suffix = val[start_idx + len(skill):]
            return f"{prefix}{placeholder}{suffix}"
            
        temp_text = re.sub(pattern, repl, temp_text)
        
    for placeholder, html_tag in placeholders.items():
        temp_text = temp_text.replace(placeholder, html_tag)
        
    return temp_text

def generate_interview_questions(missing_skills):
    questions = []
    for s in missing_skills:
        skill_key = s.lower()
        found = False
        for k, q in INTERVIEW_QUESTIONS.items():
            if k.lower() == skill_key:
                questions.append((s, q))
                found = True
                break
        if not found:
            questions.append((s, f"Can you describe your experience working with {s} or a comparable tool, and how you would apply it here?"))
    return questions
def make_gauge(score, title="Match Score"):
    label, color = score_label(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": title, "font": {"size": 14, "color": "#d1d5db"}},
        number={"suffix": "%", "font": {"size": 28, "color": "#e5e7eb"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b5563", "tickfont": {"color": "#6b7280"}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25],  "color": "rgba(239,68,68,0.12)"},
                {"range": [25, 45], "color": "rgba(249,115,22,0.12)"},
                {"range": [45, 70], "color": "rgba(245,158,11,0.12)"},
                {"range": [70, 100],"color": "rgba(16,185,129,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=200,
        font_color="#e5e7eb",
    )
    return fig

def make_score_bar_chart(results):
    names = [r["Filename"].replace(".pdf", "").replace(".docx", "") for r in results]
    scores = [r["Composite Score"] for r in results]
    colors = [score_label(s)[1] for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation='h',
        marker_color=colors,
        text=[f"{s}%" for s in scores],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Score: %{x}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.05)", color="#9ca3af"),
        yaxis=dict(color="#e5e7eb"),
        margin=dict(l=10, r=60, t=20, b=10),
        height=max(200, len(results) * 52),
        font_color="#e5e7eb",
    )
    return fig

def make_radar_chart(candidates, jd_skills):
    """Radar chart comparing skill coverage across selected candidates."""
    if not jd_skills or len(jd_skills) < 3:
        return None
    categories = list(jd_skills)[:10]  # Cap at 10 skills for readability
    fig = go.Figure()
    palette = ["#818cf8", "#34d399", "#f59e0b", "#f87171", "#a78bfa"]
    
    def hex_to_rgba(hex_str, alpha=0.15):
        h = hex_str.lstrip('#')
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    for i, cand in enumerate(candidates):
        values = [1 if skill in cand["Extracted Skills"] else 0 for skill in categories]
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=cand["Filename"].replace(".pdf", "").replace(".docx", ""),
            line_color=palette[i % len(palette)],
            fillcolor=hex_to_rgba(palette[i % len(palette)], 0.15),
            opacity=0.9,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 1],
                            ticktext=["✗", "✓"], color="#6b7280"),
            angularaxis=dict(color="#9ca3af"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e5e7eb"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=30, b=30),
        height=380,
        font_color="#e5e7eb",
    )
    return fig

def make_score_breakdown_chart(results, skill_weight_pct=60, text_weight_pct=40):
    """Grouped bar chart: skill coverage vs text similarity for each candidate."""
    names = [r["Filename"].replace(".pdf", "").replace(".docx", "") for r in results]
    skill_pcts = [r["Skill Coverage %"] for r in results]
    text_pcts  = [r["Text Similarity %"] for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"Skill Coverage ({skill_weight_pct}%)", x=names, y=skill_pcts, marker_color="#818cf8",
                         hovertemplate="%{x}<br>Skill Coverage: %{y}%<extra></extra>"))
    fig.add_trace(go.Bar(name=f"Text Similarity ({text_weight_pct}%)", x=names, y=text_pcts, marker_color="#34d399",
                         hovertemplate="%{x}<br>Text Similarity: %{y}%<extra></extra>"))
    fig.update_layout(
        barmode='group',
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#9ca3af", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(color="#9ca3af", gridcolor="rgba(255,255,255,0.04)", range=[0, 110]),
        legend=dict(font=dict(color="#e5e7eb"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=300,
        font_color="#e5e7eb",
    )
    return fig

def pills_html(skills, css_class):
    if not skills:
        return "<em style='color:#6b7280;font-size:0.82rem;'>None</em>"
    return '<div class="pill-wrap">' + \
           "".join(f'<span class="pill {css_class}">{s}</span>' for s in skills) + \
           '</div>'

RANK_ICONS = {1: "🥇", 2: "🥈", 3: "🥉"}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=72)
st.sidebar.title("Screening Engine")
st.sidebar.markdown("Configure settings and datasets.")
st.sidebar.divider()

match_mode = st.sidebar.radio(
    "Matching Algorithm",
    ["TF-IDF (Keyword Alignment)", "SBERT (Semantic Match)"],
    help="TF-IDF focuses on exact keywords. SBERT understands context and synonyms."
)
use_semantic = "SBERT" in match_mode

st.sidebar.divider()

st.sidebar.subheader("⚖️ Scoring Weights")
skill_weight_pct = st.sidebar.slider(
    "Skill Coverage Weight (%)",
    min_value=0, max_value=100, value=60, step=5,
    help="Weight of keyword/skills matching in the final composite score."
)
text_weight_pct = 100 - skill_weight_pct
st.sidebar.caption(f"Composite: **{skill_weight_pct}%** Skills + **{text_weight_pct}%** Similarity")

skill_weight = skill_weight_pct / 100.0
text_weight = text_weight_pct / 100.0

st.sidebar.divider()

dataset_path = "data/UpdatedResumeDataSet.csv"
dataset_downloaded = os.path.exists(dataset_path)
st.sidebar.subheader("Kaggle Reference Dataset")
if dataset_downloaded:
    st.sidebar.success("🟢 Dataset Active")
else:
    st.sidebar.warning("🟡 Dataset Missing")
    if st.sidebar.button("Download Dataset via API", type="primary"):
        with st.sidebar.status("Downloading...") as s:
            try:
                result = subprocess.run(["python", "download_data.py"], capture_output=True, text=True, check=True)
                s.write(result.stdout)
                if os.path.exists(dataset_path):
                    s.update(label="Downloaded!", state="complete", expanded=False)
                    st.rerun()
                else:
                    s.update(label="Failed — check logs", state="error")
                    st.sidebar.error(result.stderr)
            except Exception as e:
                s.update(label="Error", state="error")
                st.sidebar.error(str(e))

st.sidebar.divider()
st.sidebar.caption("AI Recruitment Support System • v2.0")

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>💼 AI Resume Screener & Candidate Ranker</h1>
  <p>Parse PDFs · Clean noise · Extract skills · Weighted scoring · Batch comparison · Visual analytics</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Screening Dashboard",
    "🔍 Candidate Deep Dive",
    "📊 Batch Comparison",
    "📁 Dataset Analytics",
    "🔥 Skill Heatmap",
])

# ══════════════════════════════════════════════
# TAB 1 — SCREENING DASHBOARD
# ══════════════════════════════════════════════
with tab1:
    col_jd, col_up = st.columns([1, 1])

    with col_jd:
        st.subheader("1. Job Description")
        selected_template = st.selectbox(
            "📋 Pre-fill from Job Template:",
            list(JD_TEMPLATES.keys()),
            key="template_select"
        )
        if "prev_template" not in st.session_state:
            st.session_state.prev_template = "Custom (Clear / Paste your own)"
            
        if st.session_state.template_select != st.session_state.prev_template:
            st.session_state.jd = JD_TEMPLATES[st.session_state.template_select]
            st.session_state.prev_template = st.session_state.template_select
            
        jd_input = st.text_area(
            "Paste the Job Description here:",
            value=st.session_state.jd,
            placeholder="e.g., We are looking for a Senior Data Scientist with expertise in Python, TensorFlow, AWS...",
            height=260
        )
        st.session_state.jd = jd_input

    with col_up:
        st.subheader("2. Upload Candidate Resumes")
        uploaded = st.file_uploader(
            "Drag & Drop PDF/DOCX resumes:",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )
        st.markdown(f"**Resumes queued:** `{len(uploaded) if uploaded else 0}`")

        # Scoring weight explainer
        st.markdown(f"""
        <div style='background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
                    border-radius:10px;padding:0.9rem 1.1rem;margin-top:0.8rem;'>
            <b style='color:#a5b4fc;'>Weighted Scoring Formula</b><br>
            <span style='color:#9ca3af;font-size:0.85rem;'>
                🏅 <b style='color:#818cf8;'>{skill_weight_pct}%</b> Skill Coverage &nbsp;+&nbsp;
                📝 <b style='color:#34d399;'>{text_weight_pct}%</b> Text Similarity
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    can_screen = jd_input.strip() != "" and uploaded

    if st.button("🚀 Analyze & Rank All Candidates", type="primary", use_container_width=True, disabled=not can_screen):
        results = []
        prog = st.progress(0.0)
        status = st.empty()

        cleaned_jd = clean_resume(jd_input)
        jd_skills = extract_skills(cleaned_jd)

        for i, f in enumerate(uploaded):
            status.text(f"Processing {i+1}/{len(uploaded)}: {f.name} …")
            os.makedirs("temp", exist_ok=True)
            tmp = os.path.join("temp", f.name)
            with open(tmp, "wb") as fp:
                fp.write(f.read())

            if tmp.endswith(".pdf"):
                raw = extract_text_from_pdf(tmp)
            elif tmp.endswith(".docx"):
                raw = extract_text_from_docx(tmp)
            else:
                raw = ""
                
            clean = clean_resume(raw)
            email, phone = extract_contact_info(raw)
            skills = extract_skills(clean)

            score_data = weighted_match_score(
                clean, cleaned_jd, skills, jd_skills, 
                use_semantic=use_semantic, 
                skill_weight=skill_weight, 
                text_weight=text_weight
            )

            matched  = [s for s in skills if s in jd_skills]
            missing  = [s for s in jd_skills if s not in skills]
            extra    = [s for s in skills if s not in jd_skills]

            results.append({
                "Filename":         f.name,
                "Email":            email,
                "Phone":            phone,
                "Composite Score":  score_data["composite_score"],
                "Skill Coverage %": score_data["skill_coverage_pct"],
                "Text Similarity %":score_data["text_similarity_pct"],
                "Extracted Skills": skills,
                "Matching Skills":  matched,
                "Missing Skills":   missing,
                "Extra Skills":     extra,
                "JD Skills":        jd_skills,
                "Raw Text":         raw,
                "Cleaned Text":     clean,
            })

            try: os.remove(tmp)
            except: pass
            prog.progress((i + 1) / len(uploaded))

        results = sorted(results, key=lambda x: x["Composite Score"], reverse=True)
        st.session_state.results = results
        status.text("✅ All done!")
        prog.empty()
        st.rerun()

    # ── Results table ──
    if st.session_state.results:
        results = st.session_state.results

        # Summary stats
        scores = [r["Composite Score"] for r in results]
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-value">{len(results)}</div><div class="stat-label">Candidates</div></div>
          <div class="stat-card"><div class="stat-value">{max(scores):.1f}%</div><div class="stat-label">Top Score</div></div>
          <div class="stat-card"><div class="stat-value">{np.mean(scores):.1f}%</div><div class="stat-label">Average</div></div>
          <div class="stat-card"><div class="stat-value">{sum(s>=45 for s in scores)}</div><div class="stat-label">Good+ Matches</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Score bar chart
        st.subheader("📊 Composite Score Overview")
        st.plotly_chart(make_score_bar_chart(results), use_container_width=True, key="bar_overview")

        # Score breakdown (skill vs text)
        st.subheader("📈 Score Breakdown by Component")
        st.plotly_chart(make_score_breakdown_chart(results, skill_weight_pct=skill_weight_pct, text_weight_pct=text_weight_pct), use_container_width=True, key="breakdown")

        # Ranked cards
        st.subheader("🏆 Ranked Candidates")
        for rank, r in enumerate(results, 1):
            label, color = score_label(r["Composite Score"])
            icon = RANK_ICONS.get(rank, f"#{rank}")
            fname = r['Filename']
            is_shortlisted = st.session_state.shortlist.get(fname, False)
            shortlist_badge = " ⭐" if is_shortlisted else ""
            with st.expander(
                f"{icon}  {fname}  —  "
                f"{r['Composite Score']}%  [{label}]{shortlist_badge}",
                expanded=(rank == 1)
            ):
                c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
                with c1:
                    st.plotly_chart(make_gauge(r["Composite Score"]), use_container_width=True, key=f"g_{rank}")
                    # Shortlist toggle
                    is_sl = st.checkbox(
                        "⭐ Shortlist this candidate",
                        value=st.session_state.shortlist.get(fname, False),
                        key=f"sl_{rank}"
                    )
                    st.session_state.shortlist[fname] = is_sl
                with c2:
                    st.markdown(f"**📧 Email:** `{r['Email']}`")
                    st.markdown(f"**📞 Phone:** `{r['Phone']}`")
                    st.markdown(f"**🏅 Skill Coverage:** `{r['Skill Coverage %']}%`")
                    st.markdown(f"**📝 Text Similarity:** `{r['Text Similarity %']}%`")
                    # Recruiter notes
                    st.markdown("**📝 Recruiter Notes:**")
                    note = st.text_area(
                        "Add notes:", 
                        value=st.session_state.notes.get(fname, ""),
                        height=90,
                        key=f"note_{rank}",
                        placeholder="e.g. Strong Python background, schedule for technical round...",
                        label_visibility="collapsed"
                    )
                    st.session_state.notes[fname] = note
                with c3:
                    st.markdown("**🟢 Matching Skills:**")
                    st.markdown(pills_html(r["Matching Skills"], "pill-green"), unsafe_allow_html=True)
                    st.markdown("**🔴 Missing Skills (Gap):**")
                    st.markdown(pills_html(r["Missing Skills"], "pill-red"), unsafe_allow_html=True)

        # Shortlist Summary
        shortlisted = [r for r in results if st.session_state.shortlist.get(r['Filename'], False)]
        if shortlisted:
            st.markdown(f"""
            <div style='background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);
                        border-radius:12px;padding:1rem 1.2rem;margin:1rem 0;'>
                <b style='color:#fbbf24;'>⭐ Shortlisted Candidates ({len(shortlisted)})</b>&nbsp;&nbsp;
                {' &nbsp;·&nbsp; '.join([f"<code>{r['Filename']}</code>" for r in shortlisted])}
            </div>
            """, unsafe_allow_html=True)

        # Export
        df_export = pd.DataFrame([{
            "Rank": i+1, "File": r["Filename"], "Email": r["Email"], "Phone": r["Phone"],
            "Shortlisted": "Yes" if st.session_state.shortlist.get(r['Filename'], False) else "No",
            "Recruiter Notes": st.session_state.notes.get(r['Filename'], ""),
            "Composite Score (%)": r["Composite Score"],
            "Skill Coverage (%)": r["Skill Coverage %"],
            "Text Similarity (%)": r["Text Similarity %"],
            "Matched Skills": ", ".join(r["Matching Skills"]),
            "Missing Skills": ", ".join(r["Missing Skills"]),
        } for i, r in enumerate(results)])

        col_dl, col_clr, _ = st.columns([1.5, 1, 4])
        with col_dl:
            st.download_button("📥 Export CSV", df_export.to_csv(index=False).encode(), "rankings.csv", "text/csv", use_container_width=True)
        with col_clr:
            if st.button("🧹 Clear Results", use_container_width=True):
                st.session_state.results = None
                st.session_state.shortlist = {}
                st.session_state.notes = {}
                st.rerun()
    else:
        st.info("Upload resumes and paste a Job Description, then click **Analyze & Rank** to see results.")

# ══════════════════════════════════════════════
# TAB 2 — CANDIDATE DEEP DIVE
# ══════════════════════════════════════════════
with tab2:
    if st.session_state.results:
        results = st.session_state.results
        names = [r["Filename"] for r in results]
        sel = st.selectbox("Select candidate to analyze:", names)
        cand = next(x for x in results if x["Filename"] == sel)
        label, color = score_label(cand["Composite Score"])

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown(f"### 👤 {cand['Filename']}")
            st.markdown(f"**📧** `{cand['Email']}`  &nbsp;|&nbsp;  **📞** `{cand['Phone']}`")
            st.plotly_chart(make_gauge(cand["Composite Score"], "Composite Match Score"), use_container_width=True, key="dv_gauge")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig_skill = go.Figure(go.Indicator(
                    mode="gauge+number", value=cand["Skill Coverage %"],
                    title={"text": "Skill Coverage", "font": {"size": 12, "color": "#9ca3af"}},
                    number={"suffix": "%", "font": {"size": 22}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#818cf8"},
                           "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0},
                ))
                fig_skill.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=150, margin=dict(l=10,r=10,t=30,b=5))
                st.plotly_chart(fig_skill, use_container_width=True, key="dv_skill_g")
            with col_s2:
                fig_text = go.Figure(go.Indicator(
                    mode="gauge+number", value=cand["Text Similarity %"],
                    title={"text": "Text Similarity", "font": {"size": 12, "color": "#9ca3af"}},
                    number={"suffix": "%", "font": {"size": 22}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#34d399"},
                           "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0},
                ))
                fig_text.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=150, margin=dict(l=10,r=10,t=30,b=5))
                st.plotly_chart(fig_text, use_container_width=True, key="dv_text_g")

            st.divider()
            st.markdown("##### 🟢 Matching Skills")
            st.markdown(pills_html(cand["Matching Skills"], "pill-green"), unsafe_allow_html=True)
            st.markdown("##### 🔴 Missing Skills (Skill Gap)")
            st.markdown(pills_html(cand["Missing Skills"], "pill-red"), unsafe_allow_html=True)
            st.markdown("##### 🔵 Additional Skills")
            st.markdown(pills_html(cand["Extra Skills"][:15], "pill-indigo"), unsafe_allow_html=True)

            st.divider()
            st.markdown("### 📋 Interview Prep & Feedback")
            if cand["Missing Skills"]:
                st.markdown("**Skill Gap Assessment**: The candidate lacks some skills specified in the Job Description. Here are recommended interview questions and actions:")
                q_list = generate_interview_questions(cand["Missing Skills"])
                for skill, q in q_list:
                    with st.container(border=True):
                        st.markdown(f"❓ **To assess {skill}:**")
                        st.caption(q)
                
                st.markdown(f"""
                <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
                            border-radius:10px;padding:0.9rem 1.1rem;margin-top:0.8rem;'>
                    <b style='color:#f59e0b;'>Onboarding Recommendation</b><br>
                    <span style='color:#d1d5db;font-size:0.85rem;'>
                        If hiring, consider dedicated training or pair-programming tasks targeting: 
                        <b>{', '.join(cand['Missing Skills'])}</b>.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("🎉 **Zero Skill Gap!** The candidate has all the required skills from the Job Description.")
                st.markdown("Recommended general behavioral question:")
                st.caption("Tell me about your most successful implementation using this stack, and how you ensured code quality and scalability.")

        with col_b:
            st.markdown("### 📄 Resume Text Inspector")
            mode = st.radio("View Mode", ["Cleaned Text (NLP Input)", "Raw Text"], horizontal=True, key="inspect_mode")
            content = cand["Cleaned Text"] if "Cleaned" in mode else cand["Raw Text"]

            st.divider()
            st.markdown("### 📧 Email Draft Generator")
            email_type = st.radio(
                "Generate email for:",
                ["✅ Acceptance / Interview Invite", "❌ Rejection"],
                horizontal=True, key="email_type_radio"
            )
            is_acceptance = "Acceptance" in email_type
            cand_name = cand['Filename'].replace('.pdf','').replace('.docx','').replace('_',' ').replace('-',' ').title()
            role_name = st.session_state.jd[:60].split('\n')[0].strip() or "the position"
            score = cand['Composite Score']
            matched = ', '.join(cand['Matching Skills'][:5]) or 'your relevant skills'
            missing = ', '.join(cand['Missing Skills'][:3])

            if is_acceptance:
                email_draft = f"""Subject: Interview Invitation – {role_name}

Dear {cand_name},

Thank you for your application. After reviewing your profile, we were impressed by your background — particularly your expertise in {matched}.

Your resume achieved a match score of {score}% against our requirements, placing you among our top candidates.

We would like to invite you to a technical interview at your earliest convenience. Please reply to this email with your availability for the coming week.

We look forward to speaking with you.

Best regards,
HR Team"""
            else:
                gap_note = f" While your profile shows strength in {matched}, we were looking for additional expertise in {missing}." if missing else ""
                email_draft = f"""Subject: Application Update – {role_name}

Dear {cand_name},

Thank you for taking the time to apply and for your interest in joining our team.

After careful consideration of all applications, we regret to inform you that we will not be moving forward with your application at this time.{gap_note}

We appreciate your effort and encourage you to apply for future openings that match your profile.

We wish you all the best in your job search.

Warm regards,
HR Team"""

            st.text_area(
                "Generated Email (click to edit & copy):",
                value=email_draft, height=280, key="email_output"
            )
            st.download_button(
                f"📩 Download Email Draft",
                data=email_draft,
                file_name=f"{'acceptance' if is_acceptance else 'rejection'}_{cand['Filename'].replace('.pdf','').replace('.docx','')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            highlighted = highlight_skills(content, cand["Matching Skills"], cand["Extra Skills"])
            st.markdown(f"""
            <div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); 
                        border-radius: 10px; padding: 1.2rem; height: 550px; overflow-y: auto; 
                        font-family: monospace; font-size: 0.85rem; line-height: 1.6; white-space: pre-wrap; color: #d1d5db;">
                {highlighted}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Complete the screening in the **Screening Dashboard** tab first.")

# ══════════════════════════════════════════════
# TAB 3 — BATCH COMPARISON
# ══════════════════════════════════════════════
with tab3:
    if st.session_state.results:
        results = st.session_state.results
        st.subheader("📊 Side-by-Side Batch Comparison")
        st.markdown("Select candidates to compare head-to-head.")

        all_names = [r["Filename"] for r in results]
        selected = st.multiselect(
            "Select candidates (2–5 recommended):",
            all_names,
            default=all_names[:min(3, len(all_names))]
        )

        if len(selected) < 2:
            st.warning("Select at least 2 candidates to compare.")
        else:
            sel_results = [r for r in results if r["Filename"] in selected]
            jd_skills = results[0]["JD Skills"] if results else []

            # Gauge row
            st.markdown("#### 🎯 Composite Scores")
            gauge_cols = st.columns(len(sel_results))
            for i, (col, r) in enumerate(zip(gauge_cols, sel_results)):
                with col:
                    st.plotly_chart(
                        make_gauge(r["Composite Score"], r["Filename"].replace(".pdf","")[:18]),
                        use_container_width=True, key=f"cmp_g_{i}"
                    )

            st.divider()

            # Score breakdown bar
            st.markdown("#### 📈 Score Component Breakdown")
            st.plotly_chart(make_score_breakdown_chart(sel_results, skill_weight_pct=skill_weight_pct, text_weight_pct=text_weight_pct), use_container_width=True, key="cmp_breakdown")

            st.divider()

            # Radar chart
            if jd_skills and len(jd_skills) >= 3:
                st.markdown("#### 🕸️ Skill Coverage Radar")
                radar = make_radar_chart(sel_results, jd_skills)
                if radar:
                    st.plotly_chart(radar, use_container_width=True, key="cmp_radar")
            else:
                st.info("Add more skills to the Job Description to enable the radar chart (minimum 3 JD skills needed).")

            st.divider()

            # Side-by-side table
            st.markdown("#### 📋 Head-to-Head Skills Summary")
            cols = st.columns(len(sel_results))
            for col, r in zip(cols, sel_results):
                label, color = score_label(r["Composite Score"])
                with col:
                    st.markdown(f"**{r['Filename'].replace('.pdf','').replace('.docx','')}**")
                    st.markdown(
                        f"<span class='badge' style='background:{color}22;color:{color};border:1px solid {color}44'>"
                        f"{r['Composite Score']}% · {label}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"*Skill Coverage:* **{r['Skill Coverage %']}%**")
                    st.markdown(f"*Text Similarity:* **{r['Text Similarity %']}%**")
                    st.markdown("**✅ Has:**")
                    st.markdown(pills_html(r["Matching Skills"], "pill-green"), unsafe_allow_html=True)
                    st.markdown("**❌ Missing:**")
                    st.markdown(pills_html(r["Missing Skills"], "pill-red"), unsafe_allow_html=True)

    else:
        st.info("Complete the screening in the **Screening Dashboard** tab first.")

# ══════════════════════════════════════════════
# TAB 4 — DATASET ANALYTICS
# ══════════════════════════════════════════════
with tab4:
    if dataset_downloaded:
        st.subheader("📁 Kaggle Resume Dataset Analytics")

        @st.cache_data
        def load_data():
            return pd.read_csv(dataset_path)

        df = load_data()
        cat_counts = df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        # Stat cards
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-value">{len(df)}</div><div class="stat-label">Total Resumes</div></div>
          <div class="stat-card"><div class="stat-value">{df['Category'].nunique()}</div><div class="stat-label">Job Categories</div></div>
          <div class="stat-card"><div class="stat-value">{cat_counts['Count'].max()}</div><div class="stat-label">Largest Category</div></div>
          <div class="stat-card"><div class="stat-value">{cat_counts['Count'].min()}</div><div class="stat-label">Smallest Category</div></div>
        </div>
        """, unsafe_allow_html=True)

        col_pie, col_bar = st.columns([1, 1])
        with col_pie:
            st.markdown("#### 🍩 Category Distribution")
            fig_pie = px.pie(cat_counts.head(10), names="Category", values="Count",
                             hole=0.45, color_discrete_sequence=px.colors.sequential.Purples_r)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb",
                                  legend=dict(font=dict(color="#e5e7eb"), bgcolor="rgba(0,0,0,0)"),
                                  margin=dict(l=0,r=0,t=20,b=0), height=340)
            st.plotly_chart(fig_pie, use_container_width=True, key="ds_pie")

        with col_bar:
            st.markdown("#### 📊 Resume Count per Category")
            fig_bar = px.bar(cat_counts, x="Count", y="Category", orientation='h',
                             color="Count", color_continuous_scale="purples",
                             labels={"Count": "# Resumes", "Category": ""})
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e5e7eb", coloraxis_showscale=False,
                                  margin=dict(l=0,r=20,t=20,b=0), height=340,
                                  xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                  yaxis=dict(gridcolor="rgba(255,255,255,0.03)"))
            st.plotly_chart(fig_bar, use_container_width=True, key="ds_bar")

        st.markdown("#### 📄 Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)
    else:
        st.warning("Dataset not downloaded yet. Click **Download Dataset via API** in the sidebar.")

# ══════════════════════════════════════════════
# TAB 5 — SKILL HEATMAP
# ══════════════════════════════════════════════
with tab5:
    if st.session_state.results:
        results = st.session_state.results
        st.subheader("🔥 Skill Frequency Heatmap")
        st.caption("Visualises which skills are most common and rare across all uploaded candidates.")

        # Build skill frequency matrix: candidates × skills
        all_skills_set = set()
        for r in results:
            all_skills_set.update(r["Extracted Skills"])
        all_skills = sorted(list(all_skills_set))
        cand_names = [r["Filename"].replace(".pdf","").replace(".docx","") for r in results]

        if not all_skills:
            st.info("No skills were extracted from the uploaded resumes. Try resumes with more technical content.")
        else:
            # Matrix: rows = candidates, cols = skills
            matrix = []
            for r in results:
                row = [1 if skill in r["Extracted Skills"] else 0 for skill in all_skills]
                matrix.append(row)

            # Heatmap
            fig_heat = go.Figure(go.Heatmap(
                z=matrix,
                x=all_skills,
                y=cand_names,
                colorscale=[
                    [0.0, "rgba(30,27,75,0.9)"],
                    [0.5, "rgba(79,70,229,0.6)"],
                    [1.0, "rgba(167,139,250,1.0)"]
                ],
                zmin=0, zmax=1,
                showscale=False,
                hovertemplate="<b>%{y}</b><br>Skill: %{x}<br>Present: %{z}<extra></extra>",
                xgap=2, ygap=2,
            ))
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis=dict(tickangle=-40, tickfont=dict(size=11), color="#9ca3af"),
                yaxis=dict(tickfont=dict(size=12), color="#e5e7eb"),
                margin=dict(l=10, r=10, t=20, b=120),
                height=max(280, len(results) * 56 + 140),
            )
            st.plotly_chart(fig_heat, use_container_width=True, key="skill_heatmap")

            st.divider()

            # Skill frequency bar chart
            skill_counts = {skill: sum(1 for r in results if skill in r["Extracted Skills"]) for skill in all_skills}
            skill_df = pd.DataFrame(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True), columns=["Skill", "Candidates"])
            skill_df["Rarity"] = skill_df["Candidates"].apply(
                lambda c: "Universal" if c == len(results) else ("Common" if c >= len(results)*0.6 else ("Moderate" if c >= len(results)*0.3 else "Rare"))
            )
            color_map = {"Universal": "#10b981", "Common": "#818cf8", "Moderate": "#f59e0b", "Rare": "#f87171"}

            col_hbar, col_legend = st.columns([3, 1])
            with col_hbar:
                st.markdown("#### 📊 Skill Frequency Across Candidates")
                fig_sbar = go.Figure(go.Bar(
                    x=skill_df["Candidates"],
                    y=skill_df["Skill"],
                    orientation='h',
                    marker_color=[color_map[r] for r in skill_df["Rarity"]],
                    text=[f"{c}/{len(results)}" for c in skill_df["Candidates"]],
                    textposition='outside',
                    hovertemplate="<b>%{y}</b><br>%{x} candidate(s)<extra></extra>",
                ))
                fig_sbar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=[0, len(results)+0.8], gridcolor="rgba(255,255,255,0.05)", color="#9ca3af"),
                    yaxis=dict(color="#e5e7eb"),
                    margin=dict(l=10, r=60, t=20, b=10),
                    height=max(300, len(all_skills) * 30),
                    font_color="#e5e7eb",
                )
                st.plotly_chart(fig_sbar, use_container_width=True, key="skill_freq_bar")

            with col_legend:
                st.markdown("#### Legend")
                for rarity, color in color_map.items():
                    cnt = len(skill_df[skill_df["Rarity"] == rarity])
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>"
                        f"<div style='width:14px;height:14px;border-radius:3px;background:{color};flex-shrink:0;'></div>"
                        f"<span style='font-size:0.88rem;color:#e5e7eb;'><b>{rarity}</b> ({cnt})</span></div>",
                        unsafe_allow_html=True
                    )
                st.divider()
                rare_skills = skill_df[skill_df["Rarity"] == "Rare"]["Skill"].tolist()
                if rare_skills:
                    st.markdown("**Rare Skills (standout candidates):**")
                    st.markdown(pills_html(rare_skills, "pill-red"), unsafe_allow_html=True)
                universal_skills = skill_df[skill_df["Rarity"] == "Universal"]["Skill"].tolist()
                if universal_skills:
                    st.markdown("**Universal Skills (everyone has):**")
                    st.markdown(pills_html(universal_skills, "pill-green"), unsafe_allow_html=True)
    else:
        st.info("Upload and analyze resumes in the **Screening Dashboard** tab first to see the skill heatmap.")
