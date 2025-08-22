## Weaver: Your AI Career Narrative Coach
An AI-powered resume coach built with Streamlit and Groq to provide instant, high-quality feedback on resumes.

This tool is specifically designed for **computer science students and tech professionals** looking to craft authentic and impactful career narratives that stand out in the industry.

## Key Features
- **Comprehensive Resume Analysis:** Get instant feedback on structure, content, and ATS compatibility.

- **Tailored Job Description Matching:** Receive custom bullet point suggestions to align your resume with a specific role.

- **ATS-Friendly LaTeX Template:** Access a built-in template to create a professional, machine-readable resume.

- **Built-in Security:** Includes a multi-layer defense against prompt injection attacks to ensure reliable and safe interaction.

- **Personalized Q&A:** Ask follow-up questions to refine your application strategy.

## How to Deploy (and create a shareable link)
This application is designed to be deployed on Streamlit Community Cloud.

1. ## Get a Groq API Key:

- Sign up for a free account at GroqCloud.

-  Go to the API Keys section and create a new secret key. Copy it.

2. ## Prepare Your GitHub Repository:

Ensure you have three files in your public repository:

- app.py (the main application script)

- requirements.txt (lists Python packages)

- packages.txt (for system-level dependencies like Tesseract OCR)

3. ## Deploy on Streamlit Cloud:

- Go to share.streamlit.io and click "New app".

- Select your repository.

- In the **"Advanced settings..."**, go to the Secrets section.

- Paste your Groq API key in the following format:

[groq]
api_key = "gsk_YourActualApiKeyGoesHere..."

Click **"Deploy!"**

## How to Run Locally
1. **Clone the repository.**

2. **Install Python dependencies:**

pip install -r requirements.txt

3. **Install system dependencies:**

- This app requires Tesseract OCR. Ensure it is installed and accessible in your system's PATH.

4. **Set your API Key as an environment variable:**

- **macOS/Linux**: export GROQ_API_KEY="your_api_key_here"

- **Windows:** set GROQ_API_KEY="your_api_key_here"
(Note: The current app.py is configured for deployment with Streamlit Secrets and would need a slight modification to read this local environment variable).

5. **Run the Streamlit app:**

- streamlit run app.py

## Technologies Used
- **Frontend:** Streamlit

- **Backend:** Python

- **AI/LLM:** Groq Cloud (with Llama 3.1 model)

- **Document Processing:** pdfplumber

- **OCR:** Tesseract

