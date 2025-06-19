# Weaver: Your Career Narrative

An AI-powered resume coach built with Streamlit and Ollama to provide instant, high-quality feedback on resumes. This app helps users craft authentic and impactful career narratives.

## Key Features

- **Comprehensive Resume Analysis:** Get instant feedback on structure, content, and ATS compatibility.
- **Tailored Job Description Matching:** Receive custom bullet point suggestions to align your resume with a specific role.
- **ATS-Friendly LaTeX Template:** Access a built-in template to create a professional, machine-readable resume.
- **Personalized Q&A:** Ask follow-up questions to refine your application strategy.

## How to Run Locally

1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Ensure you have [Ollama](https://ollama.com/) installed and running.
4.  Run the Streamlit app: `streamlit run app.py`

## Technologies Used

- **Frontend:** Streamlit
- **Backend:** Python
- **AI/LLM:** Ollama (with gemma3:4b model)
- **Document Processing:** pdfplumber, python-docx
- **OCR:** Tesseract