import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
import json

# 1. Load your CSV
df = pd.read_csv("intent_data.csv")
df.head()

# 2. Prepare labels
labels = sorted(df["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# Map labels to ids
df["labels"] = df["label"].map(label2id)

# 3. Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# 4. Tokenizer & model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# 6. Training setup
args = TrainingArguments(
    output_dir="intent_model",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
)

def compute_metrics(eval_pred):
    logits, y = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == y).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. Train the model
trainer.train()

# 8. Save model & tokenizer
trainer.save_model("intent_model")
tokenizer.save_pretrained("intent_model")

print("✅ Intent model saved to /content/intent_model")