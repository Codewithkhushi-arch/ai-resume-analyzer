# 📄 ResumeIQ: AI Resume Analyser

> Built a corporate-level resume screening system using Google Gemini AI + Streamlit.

This AI-powered application analyzes resumes against job descriptions, providing an enterprise-grade applicant tracking evaluation. It takes a candidate's resume (PDF) and role requirements to accurately simulate an HR/Recruiter screening process for 13+ technical roles (Software Engineering, Data Science, DevOps, AI/ML, and more).

---

## 🌟 Features
- **🎯 ATS Scoring & Job Match**: Evaluates the resume against the job description and provides an Applicant Tracking System (ATS) score with a match percentage.
- **🧠 Skills Gap Analysis**: Identifies present skills, missing keywords, and provides actionable insights to improve the candidate's profile.
- **🎤 Interview Prep**: Generates personalized behavioral and technical mock interview questions and tips based on the candidate's resume gaps and the target role.
- **📈 Career Roadmap**: Provides a customized learning path and timeframe for upskilling, along with recommended books, courses, and practice platforms.

## 🛠️ Tech Stack & Architecture

| Layer / Category | Tool / Technology | Purpose |
|------------------|-------------------|---------|
| **Core Language**| Python 3.8+ | Back-end logic and processing |
| **Web App** | Streamlit | Frontend UI framework |
| **AI Model** | Google Gemini 2.5 Flash| Core LLM for resume analysis |
| **PDF Reading** | PyPDF2 | Extracting text from PDF resumes |
| **Environment** | python-dotenv | Storing API keys securely |
| **Data** | Pandas | Tabular data handling & analytics |
| **Regex** | `re` (built-in) | Cleaning JSON responses |
| **JSON** | `json` (built-in) | Parsing AI model output |

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/resume-iq.git
   cd resume-iq
   ```

2. **Install requirements:**
   ```bash
   pip install streamlit google-generativeai PyPDF2 python-dotenv
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   *(Or simply execute `run.bat` on Windows)*

## 📸 Overview
The project incorporates a modern, dark-themed UI built with custom CSS, showcasing metrics intuitively using progress bars and scorecards to offer a seamless user experience.

---
*This project was developed to demonstrate the integration of Large Language Models (LLMs) with intuitive frontends for HR Tech and Talent Acquisition use-cases.*
