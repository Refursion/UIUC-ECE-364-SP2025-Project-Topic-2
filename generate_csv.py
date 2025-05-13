import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import os
import pandas as pd
from ProjectDataloader import TweetDataset
from tqdm import tqdm


model_chosen = 'ALBERT'
if model_chosen == 'TinyBERT':
    paths={
    'chosen_model_path' : "./model_assets/tinybert"}
    tokenizer = AutoTokenizer.from_pretrained(paths['chosen_model_path'])
elif model_chosen == 'ALBERT':
    paths = {
        'chosen_model_path' : "./model_assets/albert"}  
    tokenizer = AutoTokenizer.from_pretrained(paths['chosen_model_path'])

else:
    raise ValueError(f"Unsupported model: {model_chosen}")


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
model.eval()


test_df = pd.read_csv(
    "./data/WNUT-2020-Task-2-Dataset/WNUT-2020-Task-2-Dataset/test.tsv",
    sep='\t',
    engine='python',
    quoting=3,
    names=["Id", "Text", "Label"]
)
label_map = {"UNINFORMATIVE": 0, "INFORMATIVE": 1}
test_df["Label"] = test_df["Label"].map(label_map)
test_dataset = TweetDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        predictions.extend(preds)

inv_label_map = {0: "UNINFORMATIVE", 1: "INFORMATIVE"}
pred_labels = [inv_label_map[p] for p in predictions]

submission_df = pd.DataFrame({
    "Id": test_df["Id"].tolist(),
    "Label": pred_labels
})
submission_df.to_csv("prediction.csv", index=False)
