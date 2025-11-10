# Accountable-Chatbot

**Author:** Sankalp Bahad  
**Language / Tools:** Python, Jupyter Notebook, OCR pipeline, conversational interface  
**Overview:**  
This repository presents *Accountable Chatbot*, a project focused on building a conversational agent that does not just respond, but also logs its internal reasoning, decisions and context to enable accountability and interpretability of its dialogue interactions.

---

## Table of Contents  
1. [Motivation](#motivation)  
2. [Key Features](#key-features)  
3. [Repository Structure](#repository-structure)  
4. [Getting Started](#getting-started)  
   1. [Prerequisites](#prerequisites)  
   2. [Installation](#installation)  
   3. [Running the Demo](#running-the-demo)  
5. [Usage](#usage)  
6. [Results & Visualisations](#results-&-visualisations)  
7. [Design & Architecture](#design-&-architecture)  
8. [Future Work](#future-work)  
9. [License & Credits](#license-&-credits)  

---

## Motivation  
In current conversational-AI deployments, bots typically provide responses but rarely explain *why* they responded this way, or keep structured logs of their reasoning. This limits transparency and accountability. The Accountable Chatbot project is designed to fill that gap: it aims to **log decision points**, **maintain reasoning context**, and **provide visualisations of behaviour** so that each chat turn becomes traceable and inspectable.

---

## Key Features  
- Dialogue interface where each user-bot turn is recorded along with metadata (context, decision reason).  
- Supports **OCR input** via the `chargesheet-ocr.ipynb` notebook: e.g., ingesting a legal charge sheet image, extracting text and responding accordingly.  
- Visual graphs and result dashboards in the *Graphs* folder, aiding analysis of behaviour over time.  
- Logging folder *Results* where transcripts and metadata live for audit.  
- Modular design: the bot logic, logging module, and visualisation components are separate and can be extended or replaced.

---

## Repository Structure  
```
├─ chargesheet-ocr.ipynb # notebook for OCR + chatbot demo on legal documentation
├─ Graphs/ # visualisations of behaviour and statistics
├─ Ratings/ # logs of user ratings or bot output ratings
├─ Results/ # recorded dialogues, logs, metadata
└─ README.md # this file
```
---

## Getting Started  

### Prerequisites  
- Python 3.x  
- Jupyter Notebook / JupyterLab  
- Libraries: `pytesseract` (for OCR), `opencv-python`, `numpy`, `pandas`, `logging` module, any chatbot backend (rule-based or ML).  
- Tesseract OCR installed on the system (if using OCR notebook).

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/SankalpBahad/Accountable-Chatbot.git  
   cd Accountable-Chatbot
   ```
Install required Python packages:
```
pip install opencv-python pytesseract numpy pandas  
```
### Running the Demo
Launch Jupyter and open chargesheet-ocr.ipynb — this notebook demonstrates ingesting a scanned document, extracting text (via OCR), feeding it to the chatbot, and logging each decision.

After running the demo, inspect the Results/ folder for generated dialogue logs and the Graphs/ folder for visualisations of performance or decision trends.

### Usage
To adapt to your own domain, modify the chatbot logic (e.g., how it reasons, selects responses) and the logging module (to capture custom metadata like timestamp, rationale, user-intent classification).

Use the Results/ transcripts for downstream auditing, error-analysis, or building visual dashboards (e.g., how many times bot asked clarification, how many times it logged a reasoning step).

Use the Graphs/ folder as a reference for how you might plot metrics like “average reasoning steps per response”, “log length vs. user satisfaction”, etc.

### Results & Visualisations
The Graphs/ folder contains example charts showing evolution of bot behaviour—number of reasoning logs over time, user-rating distributions, response latency per turn.

The Ratings/ folder records feedback (if any) on bot responses — enabling you to correlate reasoning-log depth with user satisfaction.

This logging + visualisation pipeline helps in interpretability: for example, you can inspect a specific user-bot turn and see the chain of reasoning logged by the system, and then inspect whether the user was satisfied.

### Design & Architecture
Input Layer: User message or uploaded image/document (via OCR notebook).

Processing / Reasoning Layer: The chatbot decision engine produces a response and a rationale (why this response).

Logging Module: Captures: user message, extracted document text (if any), bot response, bot rationale, timestamp, metadata (dialogue turn id, session id).

Persistence Layer: Saves logs into Results/, updates running metrics in Ratings/.

Visualisation Layer: Analytical notebooks or scripts generate charts in Graphs/.
This separation enables transparency: each response is backed by a recorded rationale and easily traceable through the log files.

### Future Work
Upgrade the chatbot engine to use a fine-tuned large language model (LLM) to generate rationales automatically.

Introduce explanation interfaces so users can ask “why did you say that?” and the bot surfaces logged rationale.

Add dashboard UI (web-based) for administrators to browse logs, filter by reasoning depth, user satisfaction, flag unusual behaviour.

Extend logging metadata to include sentiment, intent classification, confidence scores, hallucination risk indicators.

Deploy the system in a dialog platform (Slack, MS Teams, Web) and collect real user-feedback to refine the accountability metrics.


