from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import os

def download_tinybert():
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    save_path = "./model_assets/tinybert"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)

def download_albert():
    model_name = "albert-base-v2"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    save_path = "./model_assets/albert"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    
if __name__ == "__main__":
    #download_tinybert()
    download_albert()
