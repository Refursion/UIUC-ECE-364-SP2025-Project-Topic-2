import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re


class TweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["Text"]
        label = self.data.iloc[idx]["Label"]
        # Remove meaningless '@USER's, which appear 3711 times in the 7001 data.
        text = re.sub(r'@USER', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Remove batch dimension
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # Load TinyBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model_assets/tinybert")

    #  load data
    df = pd.read_csv("./data/WNUT-2020-Task-2-Dataset/WNUT-2020-Task-2-Dataset/train.tsv", sep='\t')

    #  Map the labels into nums
    label_map = {"UNINFORMATIVE": 0, "INFORMATIVE": 1}
    df["Label"] = df["Label"].map(label_map)
    train_dataset = TweetDataset(df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for batch in train_loader:
        print(batch)
        break 
