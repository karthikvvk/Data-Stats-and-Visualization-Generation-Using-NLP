from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import csv

with open('train_data.csv', mode='r') as file:
    reader = csv.DictReader(file)
    data = []
    for row in reader:
        query = row['query']
        keywords = row['keywords']
        graph_requirement = row['graph_requirement']
        
        data.append({
            "query": query,
            "keywords": keywords,
            "graph_requirement": graph_requirement })

formatted_data = {
    "query": [item["query"] for item in data],
    "keywords": [item["keywords"] for item in data],
    "graph_requirement": [item["graph_requirement"] for item in data] }
dataset = Dataset.from_dict(formatted_data)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def tokenize_function(examples):
    return tokenizer(examples["query"], padding=True, truncation=True, max_length=512)

encoded_dataset = dataset.map(tokenize_function, batched=True)
encoded_dataset = encoded_dataset.map( lambda x: {"labels": [1 if label == "needed" else 0 for label in x["graph_requirement"]]}, batched=True)
train_dataset, test_dataset = encoded_dataset.train_test_split(test_size=0.2).values()
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs" )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer )
trainer.train()
model.save_pretrained("./trained_roberta")
tokenizer.save_pretrained("./trained_roberta")

