# 🛠️ Project: AI Fairness Auditor (3-Week Sprint)
**Objective:** Build a web-based tool to measure, flag, and mitigate bias in datasets and machine learning models.

---

## 📅 Week 1: Foundation & Data Profiling
*Focus: Setting up the environment and building the "Measure" capability.*

- [ ] **Day 1: Project Scaffolding**
    - Initialize a Monorepo: `/frontend` (React/Next.js) and `/backend` (FastAPI).
    - Set up Docker Compose for local development.
    - Install core ML libraries: `pandas`, `scikit-learn`, `fairlearn`, `aif360`.
- [ ] **Day 2: File Upload API**
    - Create a FastAPI endpoint to accept `.csv` or `.json` file uploads.
    - Implement a "Schema Discovery" function that returns column names to the frontend so users can select "Target" vs "Protected" attributes.
- [ ] **Day 3: Demographic Profiling (The "Measure" Phase)**
    - Write a Python service to calculate basic representation:
        - Counts and percentages for protected groups (e.g., Male vs Female).
        -     - Return a JSON object containing these distributions.
- [ ] **Day 4-5: Basic Frontend Dashboard**
    - Build an "Upload" page with a drag-and-drop zone.
    - Create a "Configuration" step where users select which column is the sensitive attribute (e.g., "Race") and which is the outcome (e.g., "Hired").
    - Render the first set of charts using `Recharts` or `Chart.js`.

---

## 📅 Week 2: The Audit Engine
*Focus: Implementing the "Flag" capability using statistical fairness metrics.*

- [ ] **Day 1: Statistical Parity & Disparate Impact**
    - Implement the **Disparate Impact Ratio**.
    - **Rule of Thumb:** If the ratio of selection rates between two groups is $< 0.8$, flag as "High Bias."
    - $$\text{Disparate Impact} = \frac{P(\hat{Y}=1 | D=\text{unprivileged})}{P(\hat{Y}=1 | D=\text{privileged})}$$
- [ ] **Day 2: Model Performance Metrics**
    - Allow users to upload a second file: "Model Predictions."
    - Calculate **False Positive Rate (FPR)** and **False Negative Rate (FNR)** per group.
    - - [ ] **Day 3: Equal Opportunity Difference**
    - Use `Fairlearn` to calculate the difference in True Positive Rates between groups.
    - A higher difference indicates the model is "missing" qualified candidates in one group more than the other.
- [ ] **Day 4-5: The "Bias Health Report" UI**
    - Create a "Scorecard" component.
    - Use color-coding: **Red** (Critical Bias), **Yellow** (Warning), **Green** (Fair).
    - Add "Explainers" (Tooltips) that define what "Disparate Impact" actually means for a non-technical user.

---

## 📅 Week 3: Mitigation & Refinement
*Focus: Implementing the "Fix" capability and polishing the UX.*

- [ ] **Day 1: Pre-processing Mitigation (Reweighing)**
    - Use `AIF360` to implement a **Reweighing** algorithm.
    - This creates a new weight for each row in the dataset to "cancel out" historical bias without deleting data.
- [ ] **Day 2: Post-processing Mitigation**
    - Implement a "Threshold Optimizer" (via `Fairlearn`) that suggests different classification cut-offs for different groups to equalize outcomes.
- [ ] **Day 3: Export & Download**
    - Create an endpoint to download the "Mitigated Dataset" (the CSV with new weights).
    - Generate a PDF "Audit Summary" that an organization could theoretically show to a regulator.
- [ ] **Day 4: UI/UX Polish & Edge Cases**
    - Handle large file uploads (streaming or chunking).
    - Add "Sample Datasets" (like the German Credit Dataset) so users can test the tool immediately without their own data.
- [ ] **Day 5: Final Demo Prep & Deployment**
    - Deploy the frontend to Vercel/Netlify.
    - Deploy the backend to Render/AWS/GCP.
    - Record a walkthrough video demonstrating the **Measure ➡️ Flag ➡️ Fix** workflow.

---

## 🧪 Testing Criteria
* **Accuracy:** Does the Disparate Impact score match manual Excel calculations?
* **Accessibility:** Does the dashboard work for people who aren't Data Scientists?
* **Privacy:** Are uploaded files deleted from the server after the session ends?
