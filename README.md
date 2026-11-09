# DataSentinel AI 🛡️
> Intelligent Data Quality, Anomaly Detection, and Natural-Language Analytics Copilot.

---

## 1. Problem Statement
Modern enterprises store millions of records across customer, sales, and transactional databases. In production pipelines, data frequently suffers from:
- Silent missing values and duplicate records.
- Schema violations and corrupted formats.
- Sudden volume spikes or drops.
- Subtle numerical and categorical **data drift** over time.

Data analysts and engineers currently identify these issues through manual spot-checks or delayed dashboard alerts. **DataSentinel AI** automates this entire lifecycle: automatically profiling datasets, detecting quality regressions and statistical anomalies, and empowering non-technical stakeholders to query data safely using plain English.

---

## 2. Target Personas
- **Data Analyst:** Needs instant answers to complex business questions without manually writing repetitive SQL.
- **Data Engineer:** Needs proactive alerts when data ingestion pipelines produce corrupted, out-of-bounds, or drifting data.
- **Business Manager:** Needs transparent, evidence-based explanations of why high-level KPIs changed.

---

## 3. Core Functional Modules
1. **Automated Data Profiler:** Ingests CSVs/Databases and computes types, null ratios, uniqueness, quantiles, and distributions.
2. **Rule-Based Quality Engine:** Validates data contracts (null constraints, uniqueness, ranges, foreign referential integrity).
3. **Statistical & ML Anomaly Detection:** Compares baseline statistical metrics (IQR / Z-Score) against Scikit-learn's `IsolationForest` on injected ground-truth faults.
4. **Data Drift Detection:** Detects distribution shifts between time windows using Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI).
5. **Safe Text-to-SQL Copilot:** Translates natural language questions to SQL with strict AST validation (read-only enforcement, blocking destructive keywords, timeouts, row limits).
6. **Root Cause Analysis (RCA):** Correlates detected anomalies, quality flags, and drift to provide structured explanations backed by data.

---

## 4. Architecture & Tech Stack (100% Free & Local)
- **Language:** Python 3.11+
- **Database Layer:** SQLite (via SQLAlchemy for seamless production portability to PostgreSQL)
- **Data & Machine Learning:** Pandas, NumPy, Scikit-learn, SciPy
- **Local GenAI / LLM:** Ollama (Llama 3.2 3B / Qwen 2.5 3B) with optional free-tier API fallback
- **Application & UI:** FastAPI (backend), Streamlit (dashboard), Plotly (visualizations)
- **Testing & Quality:** Pytest, Git

---

## 5. Development Roadmap
- [x] **Phase 1:** Setup, Architecture, and Ingestion Engine *(Completed)*
- [x] **Phase 2:** Database & Analytical Query Engine *(Completed)*
- [x] **Phase 3:** Profiling & Rule-Based Data Quality Engine *(Completed)*
- [x] **Phase 4:** ML Anomaly & Distribution Drift Detection *(Completed)*
- [x] **Phase 5:** Safe Read-Only Text-to-SQL Copilot *(Completed)*
- [x] **Phase 6:** Root Cause Analysis (RCA) Engine *(Completed)*
- [ ] **Phase 7:** FastAPI Endpoints & Streamlit Dashboard
- [ ] **Phase 8:** CI/CD, Test Suite, and Docker Packaging