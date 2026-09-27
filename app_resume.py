import streamlit as st
import os
import tempfile
import json
import joblib
import numpy as np
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import HashingVectorizer

# Page Setup
st.set_page_config(page_title="AI Resume Screening Assistant", page_icon="📄")
st.title("📄 AI Resume Screening Assistant (LangChain & RAG)")

# Simple Local Embeddings for FAISS
class SimpleFeatureEmbeddings(Embeddings):
    def __init__(self):
        self.vec = HashingVectorizer(n_features=256, alternate_sign=False)
    def embed_documents(self, texts):
        arr = self.vec.transform(texts).toarray()
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()
    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Load ML Model and Scaler (.pkl)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

model = joblib.load(model_path) if os.path.exists(model_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Sidebar Configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "API Key",
    value="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",
    type="password"
)
base_url = st.sidebar.text_input("Base URL", value="https://openrouter.ai/api/v1")
model_name = st.sidebar.text_input("Model Name", value="openrouter/free")

# Inputs
st.subheader("1. Job Description")
default_jd = """Role: Senior Data Scientist
Requirements:
- 4+ years of machine learning & Python experience
- Strong SQL, Scikit-learn, and PyTorch
- Experience with NLP and cloud deployment (AWS preferred)
"""
jd = st.text_area("Job Description", value=default_jd, height=120)

st.subheader("2. Select or Upload Resume")
sample_resumes = {
    "Alex Chen (Senior Data Scientist)": os.path.join(current_dir, "data", "resume_A_data_scientist.pdf"),
    "Priya Sharma (ML Engineer)": os.path.join(current_dir, "data", "resume_B_ml_engineer.pdf"),
    "Rahul Verma (Frontend Dev)": os.path.join(current_dir, "data", "resume_C_web_developer.pdf")
}

option = st.radio("Choose Option", ["Use Sample Resume", "Upload My PDF Resume"])
pdf_file = None

if option == "Use Sample Resume":
    choice = st.selectbox("Sample Candidate", list(sample_resumes.keys()))
    pdf_file = sample_resumes[choice]
else:
    uploaded = st.file_uploader("Upload PDF Resume", type=["pdf"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            pdf_file = tmp.name

# Evaluate Button
if st.button("Analyze Resume", type="primary"):
    if not pdf_file or not os.path.exists(pdf_file):
        st.error("Please upload or choose a resume PDF.")
    else:
        with st.spinner("Extracting text and running RAG evaluation..."):
            # Extract PDF Text
            reader = PdfReader(pdf_file)
            resume_text = "\n".join([page.extract_text() or "" for page in reader.pages])

            # Chunk and FAISS index
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = splitter.create_documents([resume_text])
            vectorstore = FAISS.from_documents(chunks, SimpleFeatureEmbeddings())
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            relevant_chunks = retriever.invoke(jd)
            context = "\n\n".join([c.page_content for c in relevant_chunks])

            # LLM Prompt
            llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                temperature=0.1,
                default_headers={"HTTP-Referer": "https://github.com", "X-Title": "Assignment"}
            )

            prompt = f"""Evaluate this candidate strictly based on the resume text for this Job Description:
Job Description:
{jd}

Resume Context:
{context}

Respond in pure JSON with keys:
"candidate_name": string
"match_score": integer (0 to 100)
"matching_skills": list of strings
"missing_skills": list of strings
"recommendation": "Shortlist", "Consider", or "Reject"
"justification": string
"""
            try:
                res = llm.invoke(prompt)
                clean_json = res.content.replace("```json", "").replace("```", "").strip()
                result = json.loads(clean_json)

                # Show Results
                st.success(f"Candidate: {result.get('candidate_name', 'Applicant')}")
                col1, col2 = st.columns(2)
                col1.metric("Match Score", f"{result.get('match_score', 0)} / 100")
                col2.metric("Recommendation", result.get('recommendation', 'N/A'))

                st.write("**Matching Skills:**", ", ".join(result.get("matching_skills", [])))
                st.write("**Missing Skills:**", ", ".join(result.get("missing_skills", [])) or "None")
                st.info(f"**Justification:** {result.get('justification', '')}")

                # ML Model Prediction (.pkl)
                if model and scaler:
                    score = float(result.get("match_score", 50))
                    skill_cnt = len(result.get("matching_skills", []))
                    feats = scaler.transform([[5.0, skill_cnt, score, 80.0]])
                    pred = model.predict(feats)[0]
                    st.write("---")
                    st.caption(f"🤖 Machine Learning Model (.pkl) Verification: {'Shortlist Recommended' if pred == 1 else 'Review Recommended'}")

            except Exception as e:
                st.error(f"Evaluation error: {e}")

if __name__ == "__main__":
    pass
