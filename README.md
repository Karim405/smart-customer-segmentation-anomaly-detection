# smart-customer-segmentation-anomaly-detection
<div align="center">

# Enterprise Customer Intelligence & Anomaly Detection System
### Dual-Engine Hybrid AI for Precision Segmentation & Behavioral Risk Profiling

<br />

<p align="center">
  <b>An enterprise-grade, high-throughput customer intelligence platform combining unsupervised Machine Learning (K-Means Clustering) with spatial density estimation (DBSCAN) to deliver real-time client segmentation, actionable marketing strategies, and behavioral anomaly detection at 120 FPS.</b>
</p>

[Live Interactive Demo](https://smart-customer-segmentation-anomaly.vercel.app) | [Executive Overview](#executive-summary) | [System Architecture](#system-architecture) | [ML Pipeline](#machine-learning-pipeline--mathematics) | [Segment Playbook](#customer-persona-playbook) | [Installation](#installation--local-setup)

---

</div>

## Executive Summary

Modern banking, financial institutions, and digital enterprises face two compounding challenges when managing customer portfolios:
1. **Generic, broad-brush customer segmentation** fails to reflect nuanced financial habits, leading to low marketing conversion, sub-optimal credit limits, and missed cross-sell opportunities.
2. **Traditional static rule engines** miss subtle behavioral drifts, sudden liquidity demands, and irregular transaction surges that precede credit defaults or fraudulent activities.

### The Solution
This platform introduces an **Enterprise Dual-Engine AI Framework** that bridges academic unsupervised learning and commercial front-office applications:
- **K-Means Clustering Engine (K=5):** Automatically organizes customer populations into distinct, statistically homogeneous behavioral personas with tailored financial playbooks.
- **DBSCAN Spatial Density Scanner (eps=1.5, min_samples=5):** Isolates high-variance outliers, noise accounts, and irregular financial behavior without requiring labeled fraud data.
- **Client-Side Edge Inference (0ms latency):** Models and mathematical transformations are pre-compiled into optimized manifest matrices, allowing client-side execution with **zero backend API overhead, 100% data privacy (GDPR / PCI-DSS compliance), and offline availability**.
- **Dual Experience Mode:**
  - **Business View:** Jargon-free, intuitive interface with clear financial metrics, automated marketing recommendations, and executive summaries.
  - **Pro Technical View:** Full diagnostic suite featuring centroid distance vectors, DBSCAN spatial density radiuses, PCA explained variances, and real-time hyperparameter tuning.

---

## System Architecture

```
+-----------------------------------------------------------------------------------------+
|                                   DATA SCIENCE PIPELINE                                 |
|  +--------------------+      +-------------------------+      +----------------------+  |
|  |   CC GENERAL CSV   | ---> | Data Cleaning & Impute  | ---> | Feature Engineering  |  |
|  |  (8,950 Customers) |      |  (Median Imputation)    |      |  (Util & Pay Ratios) |  |
|  +--------------------+      +-------------------------+      +----------------------+  |
|                                                                          |              |
|                                                                          v              |
|  +--------------------+      +-------------------------+      +----------------------+  |
|  |  model_manifest    | <--- |   StandardScaler & PCA  | <--- | Log1p Transformation |  |
|  |   (JSON Weights)   |      |   Dimensionality (2D)   |      | (Skewness Reduction) |  |
|  +--------------------+      +-------------------------+      +----------------------+  |
+------------------------------------------+----------------------------------------------+
                                           | Pre-calculated Centroids & Core Points
                                           v
+-----------------------------------------------------------------------------------------+
|                                CLIENT-SIDE EDGE AI ENGINE                               |
|  +-----------------------------------------------------------------------------------+  |
|  |                              src/lib/modelEngine.ts                               |  |
|  |  * Vectorized Log1p & Z-Scale Mapping                                             |  |
|  |  * Euclidean Distance Minimization to 5 Centroids (K-Means)                        |  |
|  |  * Nearest-Core Distance Density Scan (DBSCAN Outlier Scoring)                    |  |
|  |  * 2D PCA Linear Projection (PC1 & PC2 Coordinates)                               |  |
|  +-----------------------------------------------------------------------------------+  |
+------------------------------------------+----------------------------------------------+
                                           | Sub-millisecond Execution (<1ms)
                                           v
+-----------------------------------------------------------------------------------------+
|                               ENTERPRISE PRESENTATION LAYER                             |
|  +------------------+  +------------------+  +------------------+  +-----------------+  |
|  | Customer         |  | Segments         |  | 2D Visual        |  | Batch CSV       |  |
|  | Profiler         |  | Playbook         |  | Landscape        |  | Scanner         |  |
|  +------------------+  +------------------+  +------------------+  +-----------------+  |
|  +-----------------------------------------------------------------------------------+  |
|  | Model Tuning Studio & AI Architecture Diagnostics                                 |  |
|  | Dual Viewport: Business Mode (Simplified) <-> Pro Technical Mode (Deep Metrics)   |  |
|  | Full Bilingual Support: RTL Arabic <-> LTR English                                |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
```

---

## Core Capabilities

### 1. Interactive Real-Time Customer Profiler
- **Instant Behavioral Classification:** Adjust financial sliders (Account Balance, Credit Limit, Purchases, Cash Withdrawals, Payment Ratio) and view real-time cluster assignments updated at 120 FPS.
- **1-Click Executive Presets:** Pre-configured with distinct profiles (VIP Regular Spender, Active Revolving Shopper, Heavy Cash Borrower, Dormant Account, Unusual Activity Alert).
- **Direct Business Guidance:** Generates instant commercial recommendations (credit line upgrades, installment packages, merchant cashback, retention alerts).

### 2. Customer Persona Playbook
- Comprehensive breakdown of all 5 behavioral segments.
- Displays cohort population counts, average balances, retail spending, cash reliance, full payment compliance, and targeted marketing campaigns.

### 3. 2D Spatial Visualization (PCA Landscape)
- High-performance HTML5 Canvas rendering of **8,950 customer records** projected onto the primary principal component axes.
- Explains **56.4% cumulative variance** (PC1: 34.8% spending activity, PC2: 21.6% liquidity patterns).
- Live hover tooltips, interactive centroid tracing, and anomaly highlighting in vibrant red crosshairs.

### 4. High-Throughput Batch CSV Scanner
- Chunk-based asynchronous processing capable of ingesting and classifying thousands of customer rows client-side in seconds without freezing the UI.
- Filter by cluster or anomaly flag, search by Customer ID, inspect row-level metrics, and export fully enriched CSV files with audit-ready prediction columns.

### 5. AI Model Lab & Hyperparameter Tuning
- Adjust DBSCAN Epsilon ($\epsilon$) and density thresholds dynamically in browser memory.
- Fine-tune custom feature weighting vectors to adapt segmentation to proprietary bank priorities.
- Customize cluster persona labels, colors, and business strategies with instant simulation preview.

---

## Customer Persona Playbook

Our unsupervised K-Means algorithm ($K=5$) discovered five distinct customer personas across the portfolio:

| Cluster | Segment Persona | Population Share | Financial Characteristics | Strategic Marketing Playbook |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **High-Limit Active Borrowers** | **17.8%** (1,589) | High credit limits ($5,445), high balances ($2,585), heavy cash advance usage ($1,658). | Offer low-interest balance transfer promotions, credit line upgrades, and liquidity protection programs. |
| **1** | **Inactive / Dormant Accounts** | **12.8%** (1,150) | Negligible balance ($117), low transaction velocity ($260 purchases), minimal engagement. | Re-engagement campaigns, first-purchase cashback incentives, annual fee waivers upon card activation. |
| **2** | **Disciplined Transactors (VIP)** | **20.4%** (1,828) | High purchases ($1,150), lowest revolving debt ($181), exceptional full-payment rate (32%). | Premium shopping rewards, merchant co-branding perks, airport lounge access, tier-up loyalty incentives. |
| **3** | **Cash Advance Reliant** | **25.3%** (2,260) | High balances ($2,028), near-zero retail purchases ($17.5 avg), utilizes card as short-term liquidity loan. | Personal loan conversion packages, structured debt consolidation, flexible installment plans. |
| **4** | **Revolving Balance Shoppers** | **23.7%** (2,123) | Strong shopping activity ($1,152), frequent installment usage, carries balances month-to-month. | 0% interest installment plans, merchant discount partnerships, credit utilization SMS alerts. |
| **Alert** | **Behavioral Anomalies (Noise)** | **3.8%** (342) | Diverges significantly from all cluster densities (e.g. extreme ATM surges, purchases >$40,000). | Dedicated compliance review, credit risk audit, anti-fraud evaluation, individualized risk containment. |

---

## Machine Learning Pipeline & Mathematics

### 1. Data Cleaning & Handling Missing Values
- Missing values in `MINIMUM_PAYMENTS` and `CREDIT_LIMIT` were treated using **robust median imputation** to prevent outlier skewness.

### 2. Feature Engineering
Domain-specific financial ratios were computed:
$$\text{Credit Utilization} = \frac{\text{BALANCE}}{\text{CREDIT\_LIMIT}}$$
$$\text{Payment Ratio} = \frac{\text{PAYMENTS}}{\text{BALANCE}}$$

### 3. Logarithmic Skew Reduction
Financial variables typically exhibit heavy right-skewed power-law distributions. We apply the natural logarithm transformation $\log(1 + x)$:
$$x_{\text{transformed}} = \ln(1 + x)$$
Applied to: `BALANCE`, `PURCHASES`, `PURCHASES_TRX`, `CASH_ADVANCE`, `CASH_ADVANCE_TRX`, `PAYMENTS`.

### 4. Z-Score Standardization
To prevent high-dollar features from dominating Euclidean distance calculations:
$$z = \frac{x - \mu}{\sigma}$$
Where $\mu$ and $\sigma$ are calculated on the baseline training dataset.

### 5. Dual-Engine Clustering & Outlier Scoring
- **K-Means:** Assigns point $x$ to cluster $S_i$ minimizing inertia:
$$\arg\min_S \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2$$
- **DBSCAN:** Given neighborhood radius $\epsilon = 1.5$ and $\text{min\_samples} = 5$, identifies core points and marks any observation outside core density envelopes as an **Anomaly (Noise)**:
$$\text{dist}(x, \text{Core}) = \min_{c \in \text{CoreSamples}} ||x - c||_2$$
If $\text{dist}(x, \text{Core}) > \epsilon$, account is flagged as an anomaly.

---

## Edge AI Architecture: Privacy & Performance Benefits

```
Traditional Cloud AI Deployment:
[Client Browser] --(Transmits Sensitive PII & Financial Data)--> [Cloud API Server] --> [Latency 200-800ms + Cloud Costs]

Our Enterprise Edge Architecture:
[Client Browser] <-- Pre-compiled Model Manifest (Weights & Scalers)
  |-- 100% In-Memory Inference (<1ms execution)
  |-- Zero Financial PII leaves the user device (Full GDPR & PCI-DSS compliance)
  +-- Zero Backend GPU/Server Infrastructure Costs
```

---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Core Framework** | React 18.3 + TypeScript 5.7 | Robust, strictly typed component architecture |
| **Build & Bundling** | Vite 6.1 | Sub-second HMR and optimized production bundling |
| **Data Science / ML** | Python 3.11, Scikit-Learn, Pandas, NumPy | Data cleaning, preprocessing, K-Means & DBSCAN training |
| **Styling & Design** | TailwindCSS 3.4, Lucide Icons | Responsive modern enterprise dark-mode interface |
| **CSV Engine** | PapaParse 5.5 | Streamed asynchronous CSV parsing and generation |
| **Deployment** | Vercel Serverless Edge | Global CDN distribution and high-availability hosting |

---

## Project Directory Structure

```
Smart Customer Segmentation & Anomaly Detection/
|-- data_cleaned.csv               # Baseline cleaned credit card records
|-- data_preprocessing.ipynb       # Jupyter notebook: Data EDA & feature engineering
|-- models.ipynb                   # Jupyter notebook: K-Means, DBSCAN & PCA modeling
|-- scratch/
|   +-- export_models.py           # Production model export script (generates manifest)
|-- src/
|   |-- App.tsx                    # Main application controller & view routing
|   |-- main.tsx                   # React root entrypoint
|   |-- index.css                  # Global Tailwind styling & design tokens
|   |-- components/
|   |   |-- Header.tsx             # Enterprise navigation bar & ViewMode toggle
|   |   |-- CustomerPredictor.tsx  # Interactive real-time customer profiler
|   |   |-- ClusterExplorer.tsx    # 5 Customer persona cards & strategy playbook
|   |   |-- PcaVisualizer.tsx      # HTML5 Canvas 2D PCA spatial landscape
|   |   |-- BatchCsvUploader.tsx   # Bulk CSV upload, scan & reporting
|   |   |-- ModelTuningStudio.tsx  # Dynamic hyperparameter tuning & weighting studio
|   |   +-- ModelArchitecture.tsx  # Dual-engine system methodology documentation
|   |-- data/
|   |   +-- model_manifest.json    # Pre-calculated centroids, scalers & core points
|   +-- lib/
|       +-- modelEngine.ts         # High-performance client-side inference engine
|-- package.json                   # Dependencies & scripts
|-- tsconfig.json                  # TypeScript compiler configuration
|-- tailwind.config.js             # Tailwind design tokens
|-- vite.config.ts                 # Vite bundler configuration
+-- README.md                      # Enterprise documentation
```

---

## Installation & Local Setup

### Prerequisites
- **Node.js**: Version 18.0.0 or higher
- **npm**: Version 9.0.0 or higher
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/smart-customer-segmentation.git
cd smart-customer-segmentation
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Launch Local Development Server
```bash
npm run dev
```
Open your browser and navigate to `http://localhost:5173` (or `http://localhost:5174`).

### 4. Build for Production
```bash
npm run build
npm run preview
```

---

## Production Deployment

### Deploy to Vercel
The repository includes a ready-to-deploy `vercel.json` configuration:
```bash
npx vercel
```
Or connect your GitHub repository directly to the [Vercel Dashboard](https://vercel.com/new).

---

## Enterprise Security & Data Governance

- **Zero Data Ingestion Risk:** Customer credit records uploaded through the Batch Scanner are processed in browser worker threads and never stored or forwarded to remote endpoints.
- **Client Anonymization:** Customer IDs can be hashed or masked prior to upload with no impact on cluster classification accuracy.
- **Audit-Ready Export:** The batch export feature produces deterministic cluster IDs and anomaly probability flags for compliance archives.

---

## License & Ownership

Distributed under the **MIT License**. See `LICENSE` for more information.

Developed with precision for enterprise analytics, data-driven financial marketing, and proactive credit risk management.
