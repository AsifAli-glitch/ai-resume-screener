<h1 align="center">🧠 AI Resume Screener & Ranker</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP-spaCy%20%7C%20NLTK-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-scikit--learn-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  An intelligent, end-to-end AI-powered recruitment tool that automatically screens, scores, and ranks resumes against a job description using NLP, TF-IDF, and semantic similarity models — all inside a stunning Streamlit dashboard.
</p>

---

## 📸 Screenshots

> Upload resumes → Paste a JD → Get ranked results instantly.

| Screening Dashboard | Skill Heatmap | Email Generator |
|---|---|---|
| Ranked candidates with scores | Skill coverage across all applicants | One-click acceptance/rejection emails |

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Smart Resume Ranking** | Composite score using Skill Coverage + TF-IDF + Semantic Similarity |
| 📊 **Skill Heatmap** | Visual matrix showing which skills each candidate has/lacks |
| ⭐ **Candidate Shortlisting** | One-click shortlist with recruiter notes per candidate |
| 📧 **Email Draft Generator** | Auto-generates acceptance or rejection emails personalized per candidate |
| 🔍 **Candidate Deep Dive** | Detailed per-resume analysis with skill gap radar chart |
| 📊 **Batch Comparison** | Side-by-side radar and bar chart comparison across candidates |
| 📁 **Dataset Analytics** | Visualise the Kaggle resume dataset distribution |
| 📥 **CSV Export** | Download full rankings with shortlist status and recruiter notes |
| 🧠 **Semantic Matching** | Optional sentence-transformers for deep contextual understanding |
| 📝 **Job Templates** | Pre-filled JD templates for common roles (Data Scientist, SWE, etc.) |

---

## 🏗️ Project Structure

```
AI-Resume-Screener/
│
├── app.py                  # Main Streamlit application
├── download_data.py        # Dataset downloader (GitHub mirrors + Kaggle fallback)
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── parser.py           # PDF / DOCX text extraction
│   ├── preprocessor.py     # Text cleaning & normalization
│   ├── extractor.py        # Skill & contact extraction (NLP)
│   └── matcher.py          # Scoring engine (TF-IDF, Semantic, Composite)
│
├── tests/
│   └── test_*.py           # Unit tests
│
└── data/                   # Downloaded dataset (auto-generated, not tracked)
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AsifAli-glitch/ai-resume-screener.git
cd ai-resume-screener
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Download the dataset (optional — needed for Dataset Analytics tab)
```bash
python download_data.py
```
> The script automatically tries public GitHub mirrors first — **no Kaggle account required**.  
> If all mirrors fail, it falls back to the Kaggle API (requires `~/.kaggle/kaggle.json`).

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. 🎉

---

## 🧠 How It Works

```
Resume (PDF/DOCX)
       │
       ▼
  Text Extraction  ──►  Preprocessing  ──►  Skill Extraction (NLP)
                                                      │
Job Description  ────────────────────────────────────►│
                                                      ▼
                                           Composite Scoring Engine
                                          ┌────────────────────────┐
                                          │  Skill Coverage (40%)  │
                                          │  TF-IDF Similarity(30%)│
                                          │  Semantic Score  (30%) │
                                          └────────────────────────┘
                                                      │
                                                      ▼
                                            Ranked Results + Dashboard
```

### Scoring Formula
$$\text{Composite Score} = w_1 \cdot \text{SkillCoverage} + w_2 \cdot \text{TF-IDF} + w_3 \cdot \text{SemanticSimilarity}$$

Default weights: **Skill 40% · TF-IDF 30% · Semantic 30%** (adjustable in sidebar)

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Plotly + Custom CSS |
| **NLP** | spaCy, NLTK |
| **ML / Similarity** | scikit-learn (TF-IDF), sentence-transformers (SBERT) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Data** | pandas, numpy |
| **Dataset** | [Kaggle: gauravduttakiit/resume-dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) |

---

## 🧪 Running Tests

```bash
python -m unittest discover tests
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Built with ❤️ by **Asif Ali**

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/AsifAli-glitch)

---

<p align="center">⭐ If this project helped you, please consider giving it a star!</p>
