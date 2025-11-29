# LLMs-Powered Natural Language Query System for Healthcare

This project implements an end-to-end pipeline that allows users to query healthcare data using **plain English**.
Unlike typical NL→SQL systems, this project supports a **full workflow**:

**User question → Intent classification → SQL generation → Query execution → Table → Chart → Summary**

The system contains three major components:

### 1. **Intent Classification Model**

* A custom fine-tuned intent classifier trained using `intent_dataset.csv`
* Built on top of a pretrained **BERT (`bert-base-uncased`)** Transformer model
* Produces intent categories such as: *aggregation, comparison, trend, filter, distribution*, etc.
* Stored in the `intent_model/` folder after training
* Used to decide:

  * Which type of SQL query to generate
  * Which type of chart to visualize

### 2. **LLM-Driven SQL Generator**

* Converts natural-language questions into SQL
* Aware of the schema of `healthcare_dataset.csv`
* Uses metadata to avoid hallucinated columns
* Produces both:

  * **SQL query**
  * **Chart intent** (bar, line, pie, trend plot, etc.)

### 3. **Streamlit Web Application (`app.py`)**

* Interactive UI for entering queries
* Displays:

  * 🧠 Generated SQL
  * 📄 Clean results table
  * 📈 Automatically generated chart based on intent
  * 📝 Natural-language summary of the results
* Fully end-to-end: from user text input → visualization

---

## 📂 What This Project Contains

```text
data/
  healthcare_dataset.csv   # The main dataset users will query
  intent_dataset.csv       # Training data for intent classifier

intent_model/              # Saved fine-tuned BERT intent classification model

src/
  app.py                    # Streamlit app (core user interface)

model_training.ipynb       # Notebook for training the intent model
logs/                       # LLM + model logging
project_proposal.pdf       # Design document
requirements.txt           # Dependencies
```

---

## 🧠 Model Training (Intent Classifier)

The intent classification model is built by fine-tuning a **BERT-based Transformer (`bert-base-uncased`)** on a labeled healthcare intent dataset.

### Base Model

* **Pretrained Model:** `bert-base-uncased`
* **Architecture:** Bidirectional Transformer Encoder
* **Tokenizer:** WordPiece tokenizer (uncased)
* **Framework:** PyTorch + Hugging Face Transformers

### Training Dataset

* Source file: `intent_dataset.csv`
* Each sample contains:

  * A natural language healthcare question
  * A corresponding intent label (e.g., aggregation, filter, trend)

### Training Objective

The model is fine-tuned for a **multi-class text classification task** to learn how to map user questions to the correct analytical intent. The trained intent model directly controls:

* The structure of generated SQL queries
* The selection of visualization types (bar, line, pie, etc.)

### Training Pipeline

* Input: Natural language healthcare queries
* Tokenization: BERT tokenizer
* Model: `AutoModelForSequenceClassification`
* Loss Function: Cross-entropy loss
* Optimizer: AdamW
* Output: Trained intent classification model saved to `intent_model/`

### Purpose in the System

The fine-tuned BERT intent model enables:

* Accurate understanding of user analytical goals
* Reduced ambiguity before SQL generation
* Automatic and correct chart type selection

---

## 🔍 What Makes This Project Unique

* Combines **ML-based intent classification + LLM SQL generation**
* Full *intent-aware* NL→SQL system
* Automatic **chart selection** based on predicted intent
* Healthcare-specific dataset with meaningful real-world queries
* End-to-end **Streamlit application** included
* Reproducible model training notebook

---

## 🧠 Example Workflow

**User:**

> “Show emergency cases under age 40 with billing less than 12,000 and blood type O+ or A-”

**System:**

1. Intent model → **Aggregation + Group-by**
2. LLM → Generates SQL
3. Executor → Runs SQL on healthcare dataset
4. Chart builder → Bar chart
5. Summary → “Show emergency cases under age 40 with billing less than 12,000 and blood type O+ or A-”

---

## 🖥️ Running the Application

```bash
streamlit run app.py
```

