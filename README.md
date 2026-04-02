# ⚖️ AI Fairness Auditor (Project Aegis)

![Fairness Auditor Banner](https://capsule-render.vercel.app/api?type=waving&color=0:6A11CB,100:2575FC&height=300&section=header&text=AI%20Fairness%20Auditor&fontSize=60&animation=fadeIn&fontAlignY=35&desc=Google%20Solution%20Challenge%202026%20%7C%20Measure,%20Flag,%20and%20Fix%20Bias&descAlignY=60&descSize=20)

[![Next.js](https://img.shields.io/badge/Frontend-Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/AI-Google_Gemini_2.5-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Python](https://img.shields.io/badge/ML-Python_3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/DevOps-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

> **Ensuring Fairness and Detecting Bias in Automated Decisions.** > Computer programs increasingly make life-changing decisions about who gets a job, a bank loan, or medical care. However, if these models learn from flawed historical data, they amplify discriminatory mistakes.

## 📌 Project Overview
The **AI Fairness Auditor** is an enterprise-grade, multi-agent platform designed to thoroughly inspect datasets and machine learning models for hidden unfairness. It provides organizations with an accessible, clear way to **Measure, Flag, and Fix** harmful bias before their systems impact real people.

### 🎯 Key Features
* **📊 Deep Statistical Auditing:** Automatically calculates industry-standard fairness metrics, including *Disparate Impact* (historical bias) and *Equal Opportunity Difference* (algorithmic bias).
* **🧠 Gemini AI Ethics Consultant:** Translates dense mathematical fairness metrics into plain-English, actionable "Executive Summaries" using the Google Gemini 2.5 Flash API.
* **🛡️ Automated Bias Mitigation:** Applies mathematical transformations to scrub the influence of protected attributes (like race or gender) from datasets, allowing users to download a "safe" dataset for future model training.
* **⚡ Stateless Architecture:** Built for scale, the backend uses UUID-based temporary file storage to ensure data privacy and handle concurrent audits safely.

---

## ⚙️ Technology Stack

| Component | Tech Stack | Description |
| :--- | :--- | :--- |
| **Frontend** | Next.js, React (TS), Tailwind CSS | Responsive UI with rich data visualization using Recharts |
| **Backend** | FastAPI (Python 3.10) | High-performance asynchronous REST API |
| **Machine Learning** | Pandas, Scikit-Learn, Fairlearn | Data manipulation, bias metric calculation, and correlation removal |
| **Generative AI** | Google GenAI SDK | Gemini API acting as the interpretive Ethics Consultant |
| **DevOps** | Docker | Full application containerization for seamless deployment |

---

## 🔄 System Architecture & Workflow

The platform utilizes a highly deterministic and non-deterministic **Multi-Agent Architecture**:

1. **Data Ingestion (`/api/upload`):** Users upload a `.csv` dataset. The stateless backend assigns a temporary session `file_id`.
2. **The Math Agent (`/api/audit`):** Microsoft's `fairlearn` library calculates exact mathematical discrepancies between privileged and unprivileged groups based on user-selected target variables.
3. **The Communications Agent (`/api/audit`):** Gemini ingests the JSON output from the Math Agent and acts as a virtual Ethics Consultant, translating the metrics into accessible business intelligence.
4. **The Mitigation Agent (`/api/mitigate`):** Applies a `CorrelationRemover` to the dataset, isolating and neutralizing the bias, and returns a sanitized dataset.

---

## 🚀 Local Development Setup

### Prerequisites
* **Docker** installed on your machine.
* A valid **Google Gemini API Key**.

### 1. Running with Docker (Recommended)
This application is fully containerized for easy deployment and dependency management.

    # Build the Docker image
    docker build -t fairness-auditor-api:v1 .

    # Run the container on port 8000
    docker run -p 8000:8000 fairness-auditor-api:v1

### 2. Running Locally (Without Docker)
If you prefer to run the backend natively:

    # Install dependencies
    pip install -r requirements.txt

    # Export your API key to your environment
    export GEMINI_API_KEY="your_api_key_here"

    # Start the FastAPI server
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

---

## 🔌 API Endpoints
* **`POST /api/upload`**: Upload a `.csv` dataset (returns a session `file_id`).
* **`POST /api/audit`**: Takes the `file_id`, target variable, and protected attributes. Returns raw statistical metrics alongside the Gemini AI Executive Summary.
* **`POST /api/mitigate`**: Applies correlation removal and returns the sanitized, unbiased dataset.

---

## 📄 License
This project is licensed under the MIT License.
