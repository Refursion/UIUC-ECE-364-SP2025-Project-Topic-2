import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from ProjectDataloader import TweetDataset
from tqdm import tqdm
import os


model_chosen = "ALBERT"  #  "TinyBERT"
augmented_data_path = "./data/augment_data/augmented_train_dataset.tsv" 
output_error_path = "./data/misclassified_augment_data/misclassified_augmented_data.tsv" 
model_weight_path = "./model_assets/albert_batch_best/model_epoch_11_valacc_0.9000.pt" 
batch_size = 16


if model_chosen == "TinyBERT":
    model_dir = "./model_assets/tinybert"
elif model_chosen == "ALBERT":
    model_dir = "./model_assets/albert"
else:
    raise ValueError(f"Unsupported model: {model_chosen}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.config.hidden_size, model.config.hidden_size // 2),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.config.hidden_size // 2, 2)
)
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.to(device)
model.eval()

df = pd.read_csv(augmented_data_path, sep='\t', quoting=3)

label_map = {"UNINFORMATIVE": 0, "INFORMATIVE": 1}
inv_label_map = {0: "UNINFORMATIVE", 1: "INFORMATIVE"}

df["Label"] = df["Label"].map(label_map)

dataset = TweetDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


error_samples = []
index = 0  
with torch.no_grad():
    for batch in tqdm(loader, desc="Filtering misclassified samples"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        wrong_indices = (preds != labels).cpu().numpy()
        batch_labels = labels.cpu().numpy()

        for idx, wrong in enumerate(wrong_indices):
            if wrong:
                text_value = df.iloc[index + idx]["Text"] 
                label_value = inv_label_map[batch_labels[idx]]  
                error_samples.append({
                    "Text": text_value,
                    "Label": label_value
                })

        index += len(wrong_indices)  


os.makedirs(os.path.dirname(output_error_path), exist_ok=True)
error_df = pd.DataFrame(error_samples)
error_df.to_csv(output_error_path, sep='\t', index=False)

print(f"Saved {len(error_samples)} samples to {output_error_path}")
