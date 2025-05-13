import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import os
import pandas as pd
from ProjectDataloader import TweetDataset

model_chosen = 'ALBERT'#'TinyBERT'

if model_chosen == 'TinyBERT':
    paths={
    'best_path' : './model_assets/tinybert_batch_best',
    'save_path' : "./model_assets/tinybert_batch_training",
    'chosen_model_path' : "./model_assets/tinybert"}
    tokenizer = AutoTokenizer.from_pretrained(paths['chosen_model_path'])
elif model_chosen == 'ALBERT':
    paths = {
        'best_path' : './model_assets/albert_batch_best',
        'save_path' : "./model_assets/albert_batch_training",
        'chosen_model_path' : "./model_assets/albert"} 
    tokenizer = AutoTokenizer.from_pretrained(paths['chosen_model_path'])

else:
    raise ValueError(f"Unsupported model: {model_chosen}")
os.makedirs(paths['save_path'], exist_ok=True)
os.makedirs(paths['best_path'], exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(paths['chosen_model_path'], num_labels=2)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.config.hidden_size, model.config.hidden_size // 2),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(model.config.hidden_size // 2, 2)
)
    

model.load_state_dict(torch.load("./model_assets/Albert_Finetune/model_epoch_2_valacc_0.9010.pt"))
model.to(device)


tokenizer = AutoTokenizer.from_pretrained(paths['chosen_model_path'])

test_df = pd.read_csv(
    "./data/WNUT-2020-Task-2-Dataset/WNUT-2020-Task-2-Dataset/test.tsv",
    sep='\t',
    engine='python',
    quoting=3, 
    names=["Id", "Text", "Label"]  
)

label_map = {"UNINFORMATIVE": 0, "INFORMATIVE": 1}
test_df["Label"] = test_df["Label"].map(label_map)
test_df = test_df.dropna(subset=["Text", "Label"]).reset_index(drop=True)
test_dataset = TweetDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

model.eval()


test_correct = 0
test_total = 0


with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)


test_accuracy = test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.4f}")