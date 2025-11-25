# LLMs-Powered Natural Language Query System for Healthcare

This project implements an end-to-end pipeline that allows users to query healthcare data using **plain English**.  
Unlike typical NL→SQL systems, this project supports a **full workflow**:

**User question → Intent classification → SQL generation → Query execution → Table → Chart → Summary**

The system contains three major components:

### 1. **Intent Classification Model**
- A custom fine-tuned intent classifier (trained using `intent_dataset.csv` & `intent_dataset_1000.csv`)
- Produces categories such as: *aggregation, comparison, trend, filter, distribution*, etc.
- Stored in the `intent_model/` folder
- Used to decide which type of SQL and which chart to generate

### 2. **LLM-Driven SQL Generator**
- Converts natural-language questions into SQL
- Aware of the schema of `healthcare_dataset.csv`
- Uses metadata to avoid hallucinated columns
- Produces both:
  - **SQL query**
  - **Chart intent** (bar, line, pie, trend plot, etc.)

### 3. **Streamlit Web Application (`app.py`)**
- Interactive UI for entering queries
- Displays:
  - 🧠 Generated SQL  
  - 📄 Clean results table  
  - 📈 Automatically generated chart based on intent  
  - 📝 Natural-language summary of the results  
- Fully end-to-end: from user text input → visualization

---

## 📂 What This Project Contains
data/

healthcare_dataset.csv # The main dataset users will query

intent_dataset.csv # Training data for intent classifier

intent_model/ # Saved intent classification model

src/

app.py # Streamlit app (core user interface)

model_training.ipynb # Notebook for training intent model

logs/ # LLM + model logging

project_proposal.pdf # Design document

requirements.txt # Dependencies

---

## 🔍 What Makes This Project Unique

- Combines **ML classification + LLM SQL generation**  
- Full *intent-aware* NL→SQL system  
- Automatic **chart selection** based on intent  
- Healthcare-specific dataset with meaningful real-world queries  
- End-to-end **Streamlit application** included  
- Reproducible model training notebook  

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

