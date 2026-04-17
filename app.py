import streamlit as st
from google import genai
import PyPDF2
import json
import os
import re
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px# ============================================================
# LOAD ENV & CONFIGURE GEMINI
# ============================================================
load_dotenv()

# Safely retrieve the API key whether locally or on Streamlit Cloud
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ GOOGLE_API_KEY is not set! Please configure it in Streamlit Cloud Secrets or your local .env file.")
    st.stop()
    
client = genai.Client(api_key=api_key)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ResumeIQ — AI Resume Analyser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0f0f13; }

    section[data-testid="stSidebar"] { background-color: #13131a !important; border-right: 1px solid #1e1e2e; }
    section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }

    .hero-title { font-family: 'Syne', sans-serif; font-size: 38px; font-weight: 700; color: #ffffff; line-height: 1.2; margin-bottom: 6px; }
    .hero-sub { font-size: 15px; color: #666; margin-bottom: 28px; }

    .score-hero { background: #13131a; border: 1px solid #1e1e2e; border-radius: 20px; padding: 32px; text-align: center; margin: 20px 0; }
    .score-hero .score-num { font-family: 'Syne', sans-serif; font-size: 72px; font-weight: 700; line-height: 1; margin-bottom: 8px; }
    .score-hero .score-label { font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 2px; }

    .metric-card { background: #13131a; border: 1px solid #1e1e2e; border-radius: 14px; padding: 18px 22px; text-align: center; }
    .metric-card .m-label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .metric-card .m-value { font-size: 22px; font-weight: 600; color: #ffffff; }
    .metric-card .m-hint { font-size: 11px; margin-top: 3px; }

    .section-card { background: #13131a; border: 1px solid #1e1e2e; border-radius: 14px; padding: 22px; margin-bottom: 14px; }
    .section-card h3 { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 600; color: #ffffff; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #1e1e2e; }

    .skill-tag-green { display:inline-block; background:#0d2818; color:#4ade80; font-size:12px; padding:4px 12px; border-radius:99px; font-weight:500; border:1px solid #166534; margin:3px; }
    .skill-tag-red   { display:inline-block; background:#2d0f0f; color:#f87171; font-size:12px; padding:4px 12px; border-radius:99px; font-weight:500; border:1px solid #7f1d1d; margin:3px; }
    .skill-tag-blue  { display:inline-block; background:#0f1f2d; color:#60a5fa; font-size:12px; padding:4px 12px; border-radius:99px; font-weight:500; border:1px solid #1e3a5f; margin:3px; }
    .skill-tag-amber { display:inline-block; background:#2d1f0f; color:#fbbf24; font-size:12px; padding:4px 12px; border-radius:99px; font-weight:500; border:1px solid #7f4d1d; margin:3px; }

    .interview-q { background: #0f0f13; border-left: 3px solid #6366f1; border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 10px; }
    .interview-q .q-num { font-size: 11px; color: #6366f1; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .interview-q .q-text { font-size: 14px; color: #e8e8e8; }
    .interview-q .q-tip { font-size: 12px; color: #555; margin-top: 6px; }

    .roadmap-step { background: #0f0f13; border: 1px solid #1e1e2e; border-radius: 12px; padding: 16px 20px; margin-bottom: 10px; display: flex; align-items: flex-start; gap: 14px; }
    .roadmap-step .step-num { background: #6366f1; color: white; font-size: 12px; font-weight: 700; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px; }
    .roadmap-step .step-content .step-title { font-size: 14px; font-weight: 500; color: #ffffff; margin-bottom: 4px; }
    .roadmap-step .step-content .step-desc { font-size: 12px; color: #666; }

    .improvement-item { background: #0f0f13; border-left: 3px solid #f59e0b; border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 8px; font-size: 13px; color: #d4d4d4; }

    .stButton > button { background: #6366f1 !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 13px !important; font-size: 14px !important; font-weight: 500 !important; width: 100% !important; font-family: 'Inter', sans-serif !important; }
    .stButton > button:hover { background: #4f46e5 !important; }

    .role-badge { display: inline-block; background: #1a1a2e; color: #818cf8; font-size: 12px; padding: 4px 14px; border-radius: 99px; border: 1px solid #2e2e5e; font-weight: 500; }

    .progress-bar-bg { background: #1e1e2e; border-radius: 99px; height: 8px; margin-top: 6px; }
    .progress-bar-fill { height: 8px; border-radius: 99px; }

    .stTextArea textarea { background: #13131a !important; color: #e8e8e8 !important; border: 1px solid #1e1e2e !important; border-radius: 10px !important; }
    .stSelectbox > div > div { background: #13131a !important; color: #e8e8e8 !important; border: 1px solid #1e1e2e !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_pdf_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return None


def analyze_resume(resume_text, job_description, role):
    prompt = f"""
You are a senior HR director and technical recruiter with 15 years of experience hiring for {role} positions at top tech companies like Google, Amazon, Microsoft and Netflix.

Analyze this resume against the job description and provide a detailed corporate-level assessment.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

TARGET ROLE: {role}

Respond ONLY with a valid JSON object. No markdown, no backticks, no extra text. Just pure JSON.

{{
    "ats_score": <number 0-100>,
    "match_percentage": <number 0-100>,
    "overall_rating": "<Excellent/Good/Average/Poor>",
    "candidate_level": "<Junior/Mid-level/Senior/Principal>",
    "summary": "<2-3 sentence professional summary of the candidate>",
    "strengths": ["<strength 1>", "<strength 2>", "<strength 3>", "<strength 4>", "<strength 5>"],
    "skills_present": ["<skill 1>", "<skill 2>", "<skill 3>", "<skill 4>", "<skill 5>", "<skill 6>"],
    "skills_missing": ["<skill 1>", "<skill 2>", "<skill 3>", "<skill 4>", "<skill 5>"],
    "keywords_matched": ["<keyword 1>", "<keyword 2>", "<keyword 3>", "<keyword 4>"],
    "keywords_missing": ["<keyword 1>", "<keyword 2>", "<keyword 3>", "<keyword 4>"],
    "improvements": [
        "<specific improvement 1>",
        "<specific improvement 2>",
        "<specific improvement 3>",
        "<specific improvement 4>",
        "<specific improvement 5>"
    ],
    "interview_questions": [
        {{"question": "<technical question 1>", "tip": "<answering tip>"}},
        {{"question": "<technical question 2>", "tip": "<answering tip>"}},
        {{"question": "<behavioral question 1>", "tip": "<answering tip>"}},
        {{"question": "<technical question 3>", "tip": "<answering tip>"}},
        {{"question": "<situational question>", "tip": "<answering tip>"}}
    ],
    "roadmap": [
        {{"title": "<step 1 title>", "description": "<what to learn/do>", "duration": "<timeframe>"}},
        {{"title": "<step 2 title>", "description": "<what to learn/do>", "duration": "<timeframe>"}},
        {{"title": "<step 3 title>", "description": "<what to learn/do>", "duration": "<timeframe>"}},
        {{"title": "<step 4 title>", "description": "<what to learn/do>", "duration": "<timeframe>"}},
        {{"title": "<step 5 title>", "description": "<what to learn/do>", "duration": "<timeframe>"}}
    ],
    "hire_recommendation": "<Strong Yes/Yes/Maybe/No>",
    "hire_reason": "<one sentence reason for hire recommendation>"
}}
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        raw = response.text.strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        return json.loads(raw)
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None


def calculate_tfidf_match(resume_text, job_desc):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(cosine_sim[0][0] * 100, 1)
    except Exception as e:
        return 0.0


def create_radar_chart():
    # Demonstrates Plotly UI for portfolio purposes
    categories = ['Python/SQL', 'Machine Learning', 'Data Viz', 'Cloud/MLOps', 'Statistics']
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
          r=[85, 70, 90, 50, 80],
          theta=categories,
          fill='toself',
          name='Candidate Profile',
          line_color='#4ade80'
    ))
    fig.add_trace(go.Scatterpolar(
          r=[90, 85, 80, 85, 85],
          theta=categories,
          fill='toself',
          name='Required Skills',
          line_color='#818cf8'
    ))
    fig.update_layout(
      polar=dict(
        radialaxis=dict(visible=True, range=[0, 100], color='#888', gridcolor='#1e1e2e'),
        angularaxis=dict(color='#e8e8e8', gridcolor='#1e1e2e'),
        bgcolor='#0f0f13'
      ),
      showlegend=True,
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
      margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def score_color(score):
    if score >= 80: return "#4ade80"
    elif score >= 60: return "#fbbf24"
    elif score >= 40: return "#f97316"
    else: return "#f87171"


def progress_bar(label, value, color):
    return f"""
    <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; font-size:12px; color:#888; margin-bottom:4px;">
            <span>{label}</span>
            <span style="font-weight:600; color:#fff;">{value}%</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{value}%; background:{color};"></div>
        </div>
    </div>
    """


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 12px; border-bottom:1px solid #1e1e2e; margin-bottom:18px;'>
        <div style='font-family:"Syne",sans-serif; font-size:22px; font-weight:700; color:white;'>📄 ResumeIQ</div>
        <div style='font-size:11px; color:#555; margin-top:3px;'>AI Resume Intelligence System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Upload & Analyse",
        "📊  ATS Analysis",
        "🔬  ML & Data Analytics",
        "🧠  Skills Intelligence",
        "🎤  Interview Prep",
        "📈  Career Roadmap"
    ])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#444; line-height:2;'>
        <span style='color:#666;'>Powered by</span><br>
        Google Gemini 1.5 Flash<br>
        <span style='color:#666;'>Supports</span><br>
        13+ Technical Roles
    </div>
    """, unsafe_allow_html=True)

    if "analysis" in st.session_state:
        st.markdown("---")
        data  = st.session_state.analysis
        ats   = data.get("ats_score", 0)
        color = score_color(ats)
        st.markdown(f"""
        <div style='text-align:center; padding:12px; background:#0f0f13; border-radius:10px; border:1px solid #1e1e2e;'>
            <div style='font-size:11px; color:#555; margin-bottom:4px;'>ATS SCORE</div>
            <div style='font-family:"Syne",sans-serif; font-size:28px; font-weight:700; color:{color};'>{ats}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 1 — UPLOAD & ANALYSE
# ============================================================
if "Upload" in page:
    st.markdown('<div class="hero-title">Resume Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload your resume and get a corporate-level AI analysis in seconds</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card"><h3>📄 Upload Resume</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"], label_visibility="collapsed")
        if uploaded_file:
            st.markdown(f"""
            <div style='background:#0d2818; border:1px solid #166534; border-radius:8px; padding:10px 14px; font-size:13px; color:#4ade80;'>
                ✅ {uploaded_file.name} uploaded successfully
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><h3>🎯 Target Role</h3>', unsafe_allow_html=True)
        role = st.selectbox("Select Role", [
            "Data Scientist",
            "ML Engineer",
            "Data Analyst",
            "Web Development",
            "Full Stack Developer",
            "Software Engineer",
            "AI Engineer",
            "DevOps Engineer",
            "AWS Cloud Engineer",
            "Java Developer",
            "Python Developer",
            "MLOps Engineer",
            "DevSecOps Engineer"
        ], label_visibility="collapsed")
        st.markdown(f'<div style="margin-top:8px;"><span class="role-badge">Optimizing for: {role}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><h3>📋 Job Description</h3>', unsafe_allow_html=True)
        job_desc = st.text_area(
            "Paste job description",
            height=240,
            placeholder="Paste the job description here...\n\nExample:\nWe are looking for a Data Scientist with 2+ years experience in Python, Machine Learning, SQL...",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  Analyse Resume"):
        if not uploaded_file:
            st.error("⚠️ Please upload your resume PDF first!")
        elif not job_desc.strip():
            st.error("⚠️ Please paste the job description!")
        else:
            with st.spinner("🤖 Gemini AI is analysing your resume..."):
                resume_text = extract_pdf_text(uploaded_file)
                if not resume_text:
                    st.error("❌ Could not read PDF. Please try a different file.")
                else:
                    result = analyze_resume(resume_text, job_desc, role)
                    if result:
                        st.session_state.analysis = result
                        st.session_state.role = role
                        st.session_state.resume_text = resume_text
                        st.session_state.job_desc = job_desc
                        st.success("✅ Analysis complete! Navigate to other pages to see full results.")
                        st.balloons()

                        ats   = result.get("ats_score", 0)
                        match = result.get("match_percentage", 0)

                        c1, c2, c3, c4 = st.columns(4)
                        for col, lbl, val, hint, clr in zip(
                            [c1, c2, c3, c4],
                            ["ATS Score", "Job Match", "Rating", "Hire?"],
                            [f"{ats}/100", f"{match}%",
                             result.get("overall_rating","—"),
                             result.get("hire_recommendation","—")],
                            ["higher is better", "resume vs JD",
                             "overall assessment", "recruiter verdict"],
                            [score_color(ats), score_color(match), "#818cf8", "#4ade80"]
                        ):
                            with col:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="m-label">{lbl}</div>
                                    <div class="m-value" style="color:{clr};">{val}</div>
                                    <div class="m-hint" style="color:#555;">{hint}</div>
                                </div>""", unsafe_allow_html=True)
                    else:
                        st.error("❌ Analysis failed. Please try again.")

    if "analysis" not in st.session_state:
        st.markdown("""
        <div style='margin-top:30px; padding:32px; background:#13131a; border:1px solid #1e1e2e; border-radius:14px; text-align:center;'>
            <div style='font-size:40px; margin-bottom:12px;'>🤖</div>
            <div style='font-size:15px; color:#fff; font-weight:500; margin-bottom:6px;'>Ready to analyse your resume</div>
            <div style='font-size:13px; color:#555;'>Upload PDF + paste job description + click Analyse</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 2 — ATS ANALYSIS
# ============================================================
elif "ATS" in page:
    st.markdown('<div class="hero-title">ATS Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Applicant Tracking System score and job match breakdown</div>', unsafe_allow_html=True)

    if "analysis" not in st.session_state:
        st.warning("⚠️ Please upload and analyse your resume first on the Upload page!")
    else:
        data  = st.session_state.analysis
        role  = st.session_state.get("role", "Data Scientist")
        ats   = data.get("ats_score", 0)
        match = data.get("match_percentage", 0)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="score-hero">
                <div class="score-label">ATS Score</div>
                <div class="score-num" style="color:{score_color(ats)};">{ats}</div>
                <div style="font-size:13px; color:#555; margin-top:4px;">out of 100</div>
                <div style="margin-top:16px;">
                    {progress_bar("ATS Score", ats, score_color(ats))}
                    {progress_bar("Job Match", match, score_color(match))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="section-card" style="height:100%;">
                <h3>📋 Assessment Summary</h3>
                <div style="margin-bottom:12px;">
                    <span style="font-size:12px; color:#555;">Overall Rating</span>
                    <div style="font-size:18px; font-weight:600; color:#818cf8; margin-top:2px;">{data.get('overall_rating','—')}</div>
                </div>
                <div style="margin-bottom:12px;">
                    <span style="font-size:12px; color:#555;">Candidate Level</span>
                    <div style="font-size:18px; font-weight:600; color:#60a5fa; margin-top:2px;">{data.get('candidate_level','—')}</div>
                </div>
                <div style="margin-bottom:12px;">
                    <span style="font-size:12px; color:#555;">Hire Recommendation</span>
                    <div style="font-size:18px; font-weight:600; color:#4ade80; margin-top:2px;">{data.get('hire_recommendation','—')}</div>
                </div>
                <div style="background:#0f0f13; border-radius:8px; padding:12px; font-size:13px; color:#888; border-left:3px solid #6366f1;">
                    {data.get('hire_reason','—')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-card"><h3>✅ Keywords Matched</h3>', unsafe_allow_html=True)
            tags = "".join([f'<span class="skill-tag-green">{k}</span>' for k in data.get("keywords_matched", [])])
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="section-card"><h3>❌ Keywords Missing</h3>', unsafe_allow_html=True)
            tags = "".join([f'<span class="skill-tag-red">{k}</span>' for k in data.get("keywords_missing", [])])
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><h3>💼 Professional Summary</h3>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:14px; color:#aaa; line-height:1.7;">{data.get("summary","—")}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><h3>⭐ Key Strengths</h3>', unsafe_allow_html=True)
        for s in data.get("strengths", []):
            st.markdown(f'<div style="padding:8px 0; border-bottom:1px solid #1e1e2e; font-size:13px; color:#d4d4d4;">✦ {s}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 2.5 — ML & DATA ANALYTICS
# ============================================================
elif "ML & Data" in page:
    st.markdown('<div class="hero-title">ML & Data Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Deep dive into semantic matching and statistical analysis</div>', unsafe_allow_html=True)
    
    if "analysis" not in st.session_state:
        st.warning("⚠️ Please upload and analyse your resume first!")
    else:
        if "resume_text" in st.session_state and "job_desc" in st.session_state:
            res_text = st.session_state.resume_text
            jd_text = st.session_state.job_desc
            
            # Traditional AI (TF-IDF)
            tfidf_score = calculate_tfidf_match(res_text, jd_text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-card"><h3>📈 Statistical NLP (TF-IDF Match)</h3>', unsafe_allow_html=True)
                st.markdown(f"""
                <p style="font-size:13px; color:#aaa;">Unlike the LLM's semantic reasoning, this represents a traditional Machine Learning keyword vector matching algorithm using <code>scikit-learn</code>.</p>
                <div style="font-size:36px; font-weight:700; color:{score_color(tfidf_score)}; margin-top:10px; margin-bottom:4px;">{tfidf_score}%</div>
                <div style="font-size:11px; color:#666; text-transform:uppercase; letter-spacing:1px;">Cosine Similarity Match</div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Text Analytics
                res_words = len(res_text.split())
                jd_words = len(jd_text.split())
                
                # We can't really assume length of set words, but simple splitting is okay for EDA
                st.markdown('<div class="section-card"><h3>📊 Text Analytics (EDA)</h3>', unsafe_allow_html=True)
                df = pd.DataFrame({
                    "Context": ["Candidate Resume", "Job Description"],
                    "Word Count": [res_words, jd_words],
                    "Unique Vocab": [len(set(res_text.lower().split())), len(set(jd_text.lower().split()))]
                })
                # Streamlit dataframe with custom dark styling fallback
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown('<div class="section-card"><h3>🕸️ Skills Profile Alignment</h3>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:13px; color:#aaa; margin-bottom:10px;">A multidimensional analysis of candidate competency against role requirements mapped across key tech verticals.</p>', unsafe_allow_html=True)
            fig = create_radar_chart()
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Missing source text. Please try re-analyzing the documents.")


# ============================================================
# PAGE 3 — SKILLS INTELLIGENCE
# ============================================================
elif "Skills" in page:
    st.markdown('<div class="hero-title">Skills Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Detailed skills gap analysis for your target role</div>', unsafe_allow_html=True)

    if "analysis" not in st.session_state:
        st.warning("⚠️ Please upload and analyse your resume first!")
    else:
        data = st.session_state.analysis
        role = st.session_state.get("role", "Data Scientist")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card"><h3>✅ Skills You Have</h3>', unsafe_allow_html=True)
            skills_present = data.get("skills_present", [])
            tags = "".join([f'<span class="skill-tag-green">{s}</span>' for s in skills_present])
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:12px; color:#555; margin-top:10px;">✅ {len(skills_present)} relevant skills found</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card"><h3>❌ Skills Gap</h3>', unsafe_allow_html=True)
            skills_missing = data.get("skills_missing", [])
            tags = "".join([f'<span class="skill-tag-red">{s}</span>' for s in skills_missing])
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:12px; color:#555; margin-top:10px;">⚠️ {len(skills_missing)} skills missing for {role}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        total = len(skills_present) + len(skills_missing)
        pct   = int((len(skills_present) / total * 100)) if total > 0 else 0

        c1, c2, c3 = st.columns(3)
        for col, lbl, val, clr in zip(
            [c1, c2, c3],
            ["Skills You Have", "Skills Missing", "Skills Coverage"],
            [len(skills_present), len(skills_missing), f"{pct}%"],
            ["#4ade80", "#f87171", "#818cf8"]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="m-label">{lbl}</div>
                    <div class="m-value" style="color:{clr};">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-card"><h3>💡 Improvement Suggestions</h3>', unsafe_allow_html=True)
        for imp in data.get("improvements", []):
            st.markdown(f'<div class="improvement-item">→ {imp}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 4 — INTERVIEW PREP
# ============================================================
elif "Interview" in page:
    st.markdown('<div class="hero-title">Interview Preparation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-generated questions based on your resume and target role</div>', unsafe_allow_html=True)

    if "analysis" not in st.session_state:
        st.warning("⚠️ Please upload and analyse your resume first!")
    else:
        data = st.session_state.analysis
        role = st.session_state.get("role", "Data Scientist")

        st.markdown(f"""
        <div style='background:#13131a; border:1px solid #1e1e2e; border-radius:12px; padding:16px 20px; margin-bottom:20px;'>
            <p style='font-size:13px; color:#666; margin:0;'>
                🎯 Questions generated specifically for your resume gaps and
                <strong style='color:#818cf8;'>{role}</strong> role requirements.
            </p>
        </div>
        """, unsafe_allow_html=True)

        for i, q in enumerate(data.get("interview_questions", []), 1):
            st.markdown(f"""
            <div class="interview-q">
                <div class="q-num">Question {i}</div>
                <div class="q-text">{q.get('question','')}</div>
                <div class="q-tip">💡 Tip: {q.get('tip','')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-card"><h3>📌 General Tips for {role}</h3>', unsafe_allow_html=True)

        tips = {
            "Data Scientist": [
                "Prepare end-to-end project explanations using STAR method",
                "Be ready to explain ML model choices and trade-offs",
                "Practice coding in Python — Pandas, NumPy, Sklearn",
                "Know your statistics — hypothesis testing, p-values, distributions",
                "Be ready to discuss how you measure model performance"
            ],
            "ML Engineer": [
                "Know MLOps concepts — CI/CD for ML, model monitoring, drift detection",
                "Be ready to discuss model deployment — Docker, Flask, FastAPI",
                "Understand system design for ML at scale",
                "Practice coding ML pipelines from scratch",
                "Know cloud platforms — AWS SageMaker, GCP Vertex AI"
            ],
            "Data Analyst": [
                "Practice SQL — window functions, CTEs, complex joins",
                "Be ready to explain dashboard and visualization choices",
                "Know Excel and Tableau or Power BI in depth",
                "Practice business problem framing and metrics definition",
                "Prepare examples of insights that drove business decisions"
            ],
            "Web Development": [
                "Practice algorithmic thinking and simple DOM manipulation",
                "Be ready to explain React/Vue lifecycle or state management",
                "Understand CSS Flexbox and Grid",
                "Explain how you optimize frontend performance",
                "Know web accessibility (a11y) standards"
            ],
            "Full Stack Developer": [
                "Be ready to design a full system architecture (frontend to database)",
                "Explain your approach to RESTful vs GraphQL APIs",
                "Understand database normalization and indexing (SQL/NoSQL)",
                "Discuss handling authentication (JWT, OAuth)",
                "Practice coding simple backend end-points and connecting them"
            ],
            "Software Engineer": [
                "Master Data Structures & Algorithms (arrays, trees, graphs)",
                "Explain Object-Oriented Programming and SOLID principles",
                "Be ready to design scalable systems (System Design)",
                "Understand version control and CI/CD pipelines",
                "Discuss debugging and writing unit tests"
            ],
            "AI Engineer": [
                "Understand LLM architectures, fine-tuning, and RAG",
                "Be ready to discuss prompt engineering strategies",
                "Explain vector databases and mathematical similarity",
                "Deploying AI models to production (API integration)",
                "Discuss ethical AI and handling biases"
            ],
            "DevOps Engineer": [
                "Know Docker and Kubernetes container orchestration",
                "Explain Infrastructure as Code (Terraform, Ansible)",
                "Understand complex CI/CD pipeline structures (Jenkins, GitHub Actions)",
                "Discuss networking, load balancing, and security",
                "Know monitoring and logging tools (Prometheus, Grafana, ELK)"
            ],
            "AWS Cloud Engineer": [
                "Know core AWS services (EC2, S3, RDS, Lambda, VPC)",
                "Explain AWS IAM and security best practices",
                "Understand Serverless architecture and cost optimization",
                "Practice designing highly available cloud architectures",
                "Discuss CloudFormation or AWS CDK"
            ],
            "Java Developer": [
                "Master core Java concepts (Collections, Streams, Multithreading)",
                "Explain Spring Boot architecture and dependency injection",
                "Understand JPA/Hibernate and database mapping",
                "Be ready to write clean, testable Java code",
                "Discuss JVM memory management and garbage collection"
            ],
            "Python Developer": [
                "Master Pythonic idioms (list comprehensions, generators, decorators)",
                "Explain web frameworks like Django or FastAPI",
                "Understand async programming in Python (asyncio)",
                "Practice writing robust Pytest unit tests",
                "Discuss packaging, environments, and poetry/pip"
            ],
            "MLOps Engineer": [
                "Understand end-to-end ML lifecycles and Model Registries",
                "Explain model monitoring for data drift and concept drift",
                "Know deployment tools (Seldon, KServe, SageMaker)",
                "Bridge the gap between Data Scientists and traditional DevOps",
                "Discuss reproducible ML pipelines (Kubeflow, MLflow)"
            ],
            "DevSecOps Engineer": [
                "Explain integrating security scanning (SAST/DAST) into CI/CD",
                "Understand container security and vulnerability scanning",
                "Discuss Shift-Left security principles",
                "Know IAM, secrets management (HashiCorp Vault), and compliance",
                "Practice threat modeling for cloud infrastructure"
            ]
        }

        for tip in tips.get(role, tips["Data Scientist"]):
            st.markdown(f'<div style="padding:8px 0; border-bottom:1px solid #1e1e2e; font-size:13px; color:#aaa;">→ {tip}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 5 — CAREER ROADMAP
# ============================================================
elif "Roadmap" in page:
    st.markdown('<div class="hero-title">Career Roadmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Your personalised learning path to land the role</div>', unsafe_allow_html=True)

    if "analysis" not in st.session_state:
        st.warning("⚠️ Please upload and analyse your resume first!")
    else:
        data  = st.session_state.analysis
        role  = st.session_state.get("role", "Data Scientist")
        level = data.get("candidate_level", "Mid-level")

        st.markdown(f"""
        <div style='background:#13131a; border:1px solid #1e1e2e; border-radius:12px; padding:16px 20px; margin-bottom:20px;'>
            <div style='font-size:14px; color:#fff; font-weight:500;'>Personalised roadmap for {level} {role}</div>
            <div style='font-size:12px; color:#555; margin-top:2px;'>Based on your current skills and identified gaps</div>
        </div>
        """, unsafe_allow_html=True)

        for i, step in enumerate(data.get("roadmap", []), 1):
            st.markdown(f"""
            <div class="roadmap-step">
                <div class="step-num">{i}</div>
                <div class="step-content">
                    <div class="step-title">{step.get('title','')}</div>
                    <div class="step-desc">{step.get('description','')}</div>
                    <div style="font-size:11px; color:#6366f1; margin-top:4px;">⏱ {step.get('duration','')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        resources = {
            "Data Scientist": [
                ("📚 Courses", ["Fast.ai ML Course (free)", "Andrew Ng ML Specialization", "Kaggle Learn (free)"]),
                ("🛠️ Practice", ["Kaggle Competitions", "Build 2-3 end-to-end projects", "Contribute to open source"]),
                ("📖 Books", ["Hands-On ML — Aurélien Géron", "Python for Data Analysis — Wes McKinney"]),
            ],
            "ML Engineer": [
                ("📚 Courses", ["MLOps Specialization — DeepLearning.AI", "AWS ML Specialty", "Full Stack Deep Learning"]),
                ("🛠️ Practice", ["Deploy a model on AWS/GCP", "Build a CI/CD pipeline for ML", "Learn Docker basics"]),
                ("📖 Books", ["Designing ML Systems — Chip Huyen", "Building ML Powered Applications"]),
            ],
            "Data Analyst": [
                ("📚 Courses", ["Google Data Analytics Certificate", "SQL for Data Science", "Tableau/Power BI tutorials"]),
                ("🛠️ Practice", ["Build 3 dashboards on real datasets", "Practice SQL on LeetCode", "Create Tableau Public portfolio"]),
                ("📖 Books", ["Storytelling with Data — Cole Knaflic", "SQL Cookbook — Anthony Molinaro"]),
            ],
            "Web Development": [
                ("📚 Courses", ["The Web Developer Bootcamp (Udemy)", "FreeCodeCamp Frontend Certification"]),
                ("🛠️ Practice", ["Build a responsive portfolio", "Clone a popular website", "Frontend Mentor challenges"]),
                ("📖 Books", ["Eloquent JavaScript", "CSS in Depth"]),
            ],
            "Full Stack Developer": [
                ("📚 Courses", ["Full Stack Open (free)", "CS50's Web Programming"]),
                ("🛠️ Practice", ["Build a full CRUD application with Auth", "Deploy a MERN/PERN stack app"]),
                ("📖 Books", ["Designing Data-Intensive Applications", "Clean Architecture"]),
            ],
            "Software Engineer": [
                ("📚 Courses", ["Grokking the System Design Interview", "Algorithms Specialization (Coursera)"]),
                ("🛠️ Practice", ["Practice LeetCode (Blind 75)", "Build a command-line tool"]),
                ("📖 Books", ["Cracking the Coding Interview", "Clean Code"]),
            ],
            "AI Engineer": [
                ("📚 Courses", ["DeepLearning.AI Generative AI for Everyone", "Hugging Face NLP Course"]),
                ("🛠️ Practice", ["Build a RAG application using Langchain", "Fine-tune a small LLM model"]),
                ("📖 Books", ["Build a Large Language Model (From Scratch)"]),
            ],
            "DevOps Engineer": [
                ("📚 Courses", ["Docker Mastery (Udemy)", "Kubernetes for the Absolute Beginner"]),
                ("🛠️ Practice", ["Containerize an existing app", "Write a complete CI/CD pipeline"]),
                ("📖 Books", ["The DevOps Handbook", "Site Reliability Engineering"]),
            ],
            "AWS Cloud Engineer": [
                ("📚 Courses", ["AWS Certified Solutions Architect Course"]),
                ("🛠️ Practice", ["Host a static website on S3", "Build a serverless API with Lambda"]),
                ("📖 Books", ["AWS Cookbook"]),
            ],
            "Java Developer": [
                ("📚 Courses", ["Spring Boot Masterclass", "Java Programming Masterclass"]),
                ("🛠️ Practice", ["Build a robust REST API using Spring Boot", "Implement Spring Security"]),
                ("📖 Books", ["Effective Java", "Spring in Action"]),
            ],
            "Python Developer": [
                ("📚 Courses", ["100 Days of Code: Python Bootcamp", "Real Python Learning Paths"]),
                ("🛠️ Practice", ["Build a high-performance Web API using FastAPI", "Write a web scraper"]),
                ("📖 Books", ["Fluent Python", "Python Crash Course"]),
            ],
            "MLOps Engineer": [
                ("📚 Courses", ["Made With ML (MLOps)", "DeepLearning.AI MLOps Specialization"]),
                ("🛠️ Practice", ["Deploy an ML model using Docker and FastAPI", "Set up MLflow"]),
                ("📖 Books", ["Introducing MLOps", "Machine Learning Engineering"]),
            ],
            "DevSecOps Engineer": [
                ("📚 Courses", ["Practical DevSecOps Certification", "AWS Certified Security"]),
                ("🛠️ Practice", ["Integrate SonarQube & Trivy into a CI pipeline", "Perform a security audit"]),
                ("📖 Books", ["Alice and Bob Learn Application Security", "Securing DevOps"]),
            ]
        }

        st.markdown('<div class="section-card"><h3>📚 Recommended Resources</h3>', unsafe_allow_html=True)
        for category, items in resources.get(role, resources["Data Scientist"]):
            st.markdown(f'<p style="font-size:13px; font-weight:500; color:#818cf8; margin-bottom:6px;">{category}</p>', unsafe_allow_html=True)
            for item in items:
                st.markdown(f'<div style="font-size:13px; color:#aaa; padding:4px 0 4px 12px; border-left:2px solid #1e1e2e;">→ {item}</div>', unsafe_allow_html=True)
            st.markdown('<div style="margin-bottom:12px;"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
