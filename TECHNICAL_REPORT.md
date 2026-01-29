# VeriFact: Automated Neural Fact-Checking System
# Technical Implementation Report

## 1. Introduction

### 1.1 Problem Statement
The rapid dissemination of information across digital platforms has exacerbated the spread of misinformation. Manual verification of claims is labor-intensive and cannot scale to meet the volume of modern content creation. There is a critical need for automated systems capable of verifying textual claims against credible external evidence in near real-time.

### 1.2 Motivation
This project, **VeriFact**, aims to bridge the gap between information retrieval and semantic reasoning. Instead of relying on static databases of previously debunked claims, the system implements a dynamic verification pipeline that retrieves live evidence from the web and uses Natural Language Inference (NLI) to determine veracity.

### 1.3 Scope and Boundaries
The system is designed as an **inference-only** pipeline. It does not train new foundation models but rather orchestrates pre-trained Transformer models to perform specific reasoning tasks. Its scope is limited to:
- Validating textual claims against English-language web sources.
- Proving provenance and semantic alignment of evidence.
- deterministically aggregating logic-based signals into a verdict.
It does not verify image/video content (deepfakes) or generate new knowledge beyond what is retrieved.

---

## 2. System Overview

### 2.1 High-Level Description
VeriFact is a modular, retrieval-augmented fact-checking API. It accepts natural language claims, retrieves relevant documents from the open web, filters content using semantic similarity, and analyzes the stance of the retrieved content relative to the claim. The final output is a structured verdict ("Likely True", "Likely False", etc.) supported by a confidence score and a transparent audit trail of evidence.

### 2.2 Design Philosophy
The system prioritizes **provenance** and **explainability** over generative capabilities. Unlike Large Language Models (LLMs) which may hallucinate facts, VeriFact functions as a neural search-and-verify engine. Every component of the verdict is traceable to a specific source URL and a specific sentence within that source.

### 2.3 Data Flow
1.  **Input Ingestion:** Raw text/URL is received via API.
2.  **Claim Extraction:** Core assertion is isolated from input noise.
3.  **Information Retrieval:** Search queries are generated and executed against external APIs.
4.  **Semantic Filtering:** Irrelevant content is discarded based on vector similarity.
5.  **Inference:** Survivor content is analyzed for stance (Support/Refute).
6.  **Aggregation:** Signals are weighted and fused into a final score.

---

## 3. Architecture and System Design

### 3.1 Architectural Components
The system follows a layered architecture:
-   **Interface Layer:** Flask REST API handling request lifecycle and validation.
-   **Controller Layer:** Orchestrates the pipeline execution.
-   **Core Services:** Specialized modules for NLP tasks (`claim_extractor`, `stance_detector`, etc.).
-   **External Interfaces:** Wrappers for Search APIs (Tavily, DuckDuckGo) and Model Loaders.

### 3.2 Interaction and Control Flow
Execution is synchronous but parallelized at the I/O layer. The main thread accepts a request and invokes the pipeline. The **Evidence Aggregation** phase utilizes a `ThreadPoolExecutor` to perform concurrent web scraping and processing of multiple search results, significantly reducing total latency.

### 3.3 Failure Handling
-   **Search Redundancy:** The search module implements a fallback chain (Tavily $\rightarrow$ Brave $\rightarrow$ DuckDuckGo) to ensure continuity if primary APIs fail or rate-limit.
-   **Graceful Degradation:** If specific URLs fail to scrape, they are logged and skipped without halting the pipeline.
-   **Safety Defaults:** If no evidence is found, the system defaults to an "UNVERIFIED" state rather than guessing.

---

## 4. Implementation Details

### 4.1 Repository Structure
The codebase is organized as follows:
-   `app_flask.py`: Application entry point and API definition.
-   `app/core/`: Functional logic.
    -   `claim_extractor.py`: Input normalization using Spacy/KeyBERT.
    -   `query_generator.py`: Query expansion logic.
    -   `web_search.py`: Search API integration.
    -   `scraper.py`: `trafilatura`-based HTML text extraction.
    -   `embedder.py`: SBERT embedding logic.
    -   `stance_detector.py`: Zero-shot classification pipeline.
    -   `verdict_engine.py`: Scoring algorithms.
-   `Dockerfile`: Container definition.
-   `.github/workflows/deploy.yml`: CI/CD configuration.

### 4.2 Entry Points
The application runs via a **Gunicorn** WSGI server in production. The `app_flask.py` module initializes the Flask app, loads heavy ML models into global memory to amortize startup cost, and exposes the `/api/check` endpoint.

---

## 5. NLP and Machine Learning Pipeline

### 5.1 Claim Processing
Raw inputs are cleaned to normalize whitespace. If a URL is provided, `trafilatura` extracts the main text body. A rule-based heuristic combined with **KeyBERT** extracts the first meaningful sentence as the primary claim, provided it meets length constraints (>20 chars).

### 5.2 Query Generation
To mitigate "keyword myopia," `spacy` Named Entity Recognition (NER) identifies entities ($E$) in the claim. The system generates 5-10 query variations, including:
-   The exact claim ($C$).
-   Verification suffixes: "$C$ fact check", "$C$ hoax".
-   Entity-centric queries: "$E$ controversy", "$E$ news verification".

### 5.3 Evidence Retrieval and Cleaning
Retrieved URLs are scraped using `trafilatura`, which strips boilerplate (ads, navbars). The text is segmented into sentences.

### 5.4 Embedding Generation
The system utilizes **Sentence-BERT (SBERT)** to map the claim ($C$) and every candidate sentence ($S_i$) into a 768-dimensional vector space. Cosine similarity is computed:
$$Similarity(C, S_i) = \frac{C \cdot S_i}{\|C\| \|S_i\|}$$
Only the sentence with the highest similarity score from each document is retained for stance analysis, acting as a relevance filter.

### 5.5 Stance Detection Logic
The "best matching sentence" is paired with the claim and passed to a **Cross-Encoder** style NLI model. The model predicts the logical relationship:
-   **Entailment:** The evidence supports the claim.
-   **Contradiction:** The evidence refutes the claim.
-   **Neutral:** The evidence discusses the topic but is inconclusive.

---

## 6. Model Selection and Usage

### 6.1 Models Used
1.  **Sentence-BERT (`all-mpnet-base-v2`):** Used for semantic search. Chosen for its state-of-the-art performance on semantic textual similarity (STS) benchmarks, offering a balance between accuracy and computational cost (~420MB).
2.  **DeBERTa-v3 (`moritzlaurer/deberta-v3-base-zeroshot-v2.0`):** Used for zero-shot stance detection. It treats the problem as NLI, where the claim is the hypothesis.
3.  **Spacy (`en_core_web_sm`):** Lightweight CPU-optimized model for tokenization and NER.

### 6.2 Inference Methodology
The system uses **Zero-Shot Classification** pipelines. This is a crucial design choice: explicit training on "fake news" datasets leads to rapid obsolescence as topics shift. Zero-shot models leverage generalized linguistic reasoning, allowing the system to verify claims on novel topics without retraining.

### 6.3 Inference Trade-offs
-   **Latency:** Transformer inference on CPU is computationally expensive. The system manages this by limiting input length and utilizing threading for web I/O to mask some latency.
-   **Throughput:** Single-instance throughput is limited. Production scaling would require GPU acceleration or horizontal scaling.

---

## 7. Evidence Aggregation and Verdict Logic

### 7.1 Evidence Scoring
Each piece of evidence $E$ is assigned a scalar score based on three factors:
1.  **Similarity ($Sim$):** How relevant the text is (0.0 - 1.0).
2.  **Stance Confidence ($Conf$):** How sure the model is of the relationship (0.0 - 1.0).
3.  **Direction ($Dir$):** +1 (Support), -1 (Refute), 0 (Neutral).

### 7.2 Credibility Weighting
The `source_scorer` module applies a domain-based weight ($W_{src}$) to each evidence source:
-   **Tier 1 (High Trust):** Reuters, AP, BBC, Gov/Edu domains ($W \approx 1.5$).
-   **Tier 2 (Standard):** General news ($W \approx 1.0$).
-   **Tier 3 (Low Trust):** Social Media, Forums ($W \approx 0.5$).

### 7.3 Verdict Computation
The final **Net Score** is the sum of weighted evidence signals:
$$NetScore = \sum (Sim_i \times Conf_i \times Dir_i \times W_{src, i})$$

### 7.4 Verdict Thresholds
The Net Score is mapped to a discrete verdict:
-   **Likely True:** Score $> 0.4$
-   **Likely False:** Score $< -0.4$
-   **Mixed / Misleading:** $-0.4 \le Score \le 0.4$
-   **Unverified:** No evidence found.

---

## 8. API Design

### 8.1 Endpoint Definitions
-   `GET /api/health`: Provides system status, uptime, and basic metrics.
-   `POST /api/check`: The primary analysis endpoint.

### 8.2 Request/Response Contract
**Request:**
```json
{
  "claim": "string",
  "url": "optional string",
  "max_results": "int (1-10)"
}
```

**Response:**
```json
{
  "verdict": "LIKELY TRUE",
  "confidence": 0.85,
  "net_score": 1.25,
  "evidences": [ ... ],
  "explanation": { ... }
}
```

### 8.3 Error Handling
All exceptions are caught at the controller level. Global error handlers standardize responses into JSON format (500/400/429), ensuring the API never exposes raw stack traces to the client while logging details for debugging.

---

## 9. Deployment and Infrastructure

### 9.1 Containerization
The application is packaged in a **Docker** container based on `python:3.10-slim`. The base image is minimized for security and size. A non-root user (`appuser`) is created to execute the application, adhering to security best practices.

### 9.2 Runtime Environment
-   **Server:** AWS EC2 (Ubuntu).
-   **Process Manager:** Gunicorn with `gthread` workers (2 workers, 2 threads per worker) to handle concurrent I/O requests alongside blocking CPU inference.
-   **Timeout:** Configured to 300 seconds to accommodate worst-case CPU inference times.

### 9.3 CI/CD Workflow
GitHub Actions orchestrates the deployment:
1.  **Test:** Runs `pytest` suite.
2.  **Build:** Builds Docker image and pushes to DockerHub.
3.  **Deploy:** SSHs into EC2 instance, performs disk cleanup (pruning images/containers), pulls the new image, and restarts the container.

---

## 10. Limitations

### 10.1 Technical Limitations
-   **CPU Inference Latency:** Running heavy Transformers (DeBERTa/SBERT) on standard CPUs results in high latency (30s-60s per request).
-   **Context Window:** The system extracts single "best" sentences. Complex claims requiring multi-sentence reasoning or cross-document context may be misclassified.

### 10.2 Model Limitations
-   **Zero-Shot Fallibility:** While robust, zero-shot models can misinterpret sarcasm or subtle nuances in language.
-   **Language Support:** The current pipeline is optimized for English; performance on other languages is undefined.

### 10.3 Data Limitations
-   **Source Reputation:** Domain credibility is based on a static lookup table (`source_scorer.py`). New or unlisted high-quality sources receive a default weight, while unlisted low-quality sources are not penalized.

---

## 11. Conclusion

VeriFact implements a robust, transparent, and extendable architecture for automated fact-checking. By combining modern Neural Information Retrieval (SBERT) with Natural Language Inference (DeBERTa), it moves beyond simple keyword matching to perform true semantic verification. The inference-only design ensures strict adherence to retrieved evidence, minimizing hallucination risks and providing users with a verifiable audit trail for every verdict.
