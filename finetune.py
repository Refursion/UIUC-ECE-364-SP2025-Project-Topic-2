import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from ProjectDataloader import TweetDataset
import random
import numpy as np

# 1. Fix random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed = 364
set_seed(seed)


class FinetuneTrainer:
    def __init__(self, model, train_loader, val_loader, tokenizer,paths, optimizer_class, loss_fn, scheduler_class, device, model_chosen = 'ALBERT'):
        
        self.model_chosen = model_chosen
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.paths = paths
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler_class = scheduler_class
        
        self.hyperparams = {}
        self.train_history = []
        self.val_history = []

        self.optimizer = None
        
    def set_hyperparams(self, lr=5e-5,head_lr=1e-3, num_epochs=20, save_interval=5, unfreeze_every=3):
        
        self.hyperparams['lr'] = lr
        self.hyperparams['head_lr'] = head_lr
        self.hyperparams['num_epochs'] = num_epochs
        self.hyperparams['save_interval'] = save_interval
        self.hyperparams['unfreeze_every'] = unfreeze_every
        
    def train(self):
        self.model.to(self.device)
        if self.model_chosen == "TinyBERT":
            encoder = self.model.bert
        elif self.model_chosen == "ALBERT":
            encoder = self.model.albert
        else:
            raise ValueError("Unsupported model.")

        self.optimizer = self.optimizer_class([
            {'params': encoder.parameters(), 'lr': self.hyperparams['lr']},
            {'params': self.model.classifier.parameters(), 'lr': self.hyperparams['head_lr']}
        ])
        self.scheduler = self.scheduler_class(self.optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        num_epochs = self.hyperparams['num_epochs']
        save_interval = self.hyperparams['save_interval']
       
        hyperparams_path = os.path.join(self.paths['save_path'], "hyperparams.json")
        if not os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'w') as f:
                json.dump(self.hyperparams, f, indent=4)
            print(f"Hyperparameters saved to {hyperparams_path}")
       
        best_valid_acc=0.0
        early_stop_patience = 4
        early_stop_counter = 0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.loss_fn(logits, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            avg_loss = total_loss / len(self.train_loader)
            train_accuracy = total_correct / total_samples
            
            self.model.eval()  
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)

                    valid_correct += (preds == labels).sum().item()
                    valid_total += labels.size(0)

            valid_accuracy = valid_correct / valid_total
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Valid Acc: {valid_accuracy:.4f}")
         
            if valid_accuracy > best_valid_acc:
                best_valid_acc = valid_accuracy
                early_stop_counter = 0  
                print(f"New best model found at epoch {epoch+1}")
               
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter} consecutive epoch(s).")

            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            if (epoch + 1) % save_interval == 0:
                save_name = f"model_epoch_{epoch+1}_valacc_{valid_accuracy:.4f}.pt"
                save_path = os.path.join(self.paths['save_path'], save_name)
                torch.save(self.model.state_dict(), save_path)
                print(f"Checkpoint saved at {save_path}")
            
            # Dynamic lr
            self.scheduler.step(valid_accuracy)
            for param_group in self.optimizer.param_groups:
                print(f"Current learning rate: {param_group['lr']}")
                
            self.train_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': train_accuracy
            })
            self.val_history.append({
                'epoch': epoch + 1,
                'accuracy': valid_accuracy
            })
    
    def summarize(self):
        
        epochs = [record['epoch'] for record in self.train_history]
        train_losses = [record['loss'] for record in self.train_history]
        train_accuracies = [record['accuracy'] for record in self.train_history]
        val_accuracies = [record['accuracy'] for record in self.val_history]
        
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, marker='o', label='Train Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, marker='o', label='Train Accuracy')
        plt.plot(epochs, val_accuracies, marker='x', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        fig_path = os.path.join(self.paths['save_path'], 'training_summary.png')
        plt.savefig(fig_path)
        print(f"Training summary plot saved to {fig_path}")
        
        plt.show()

def create_mixed_loader(hard_df, train_df, tokenizer, hard_to_normal_ratio=1.0, batch_size=16, shuffle=True, seed=seed):
    """
    Create a mixed DataLoader by combining hard samples and original samples in a specified ratio.

    Parameters:
    - hard_df: DataFrame containing augmented hard (difficult) samples
    - train_df: DataFrame of the original training set (large)
    - tokenizer: Preloaded tokenizer
    - hard_to_normal_ratio: Ratio of hard samples to normal samples (e.g., 1.0 means 1:1)
    - batch_size: Batch size for the DataLoader
    - shuffle: Whether to shuffle the data
    - seed: Random seed for reproducibility

    Returns:
    - mixed_loader: DataLoader for training
    """
    hard_num = len(hard_df)
    normal_num = int(hard_num * hard_to_normal_ratio)

    sampled_train_df = train_df.sample(n=normal_num, random_state=seed)
  
    mixed_df = pd.concat([hard_df, sampled_train_df], ignore_index=True).sample(frac=1.0, random_state=seed)

    mixed_dataset = TweetDataset(mixed_df, tokenizer)

    g = torch.Generator()
    g.manual_seed(seed)
    mixed_loader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=shuffle, worker_init_fn=lambda worker_id: torch.manual_seed(seed), generator=g)

    return mixed_loader


model_chosen = 'ALBERT' #'TinyBERT'

if model_chosen == 'TinyBERT':  
    paths={
        'save_path' : "./model_assets/tinybert_batch_training",
        'chosen_model_path' : "./model_assets/tinybert",
        'weight_path': ""}
    tokenizer = AutoTokenizer.from_pretrained(paths['chosen_model_path'])
elif model_chosen == 'ALBERT':
    paths = {
        'save_path' : "./model_assets/Albert_Finetune",
        'chosen_model_path' : "./model_assets/albert",
        'weight_path': "./model_assets/albert_batch_best/model_epoch_11_valacc_0.9000.pt"} 
    tokenizer = AutoTokenizer.from_pretrained(paths['chosen_model_path'])

else:
    raise ValueError(f"Unsupported model: {model_chosen}")
os.makedirs(paths['save_path'], exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(paths['chosen_model_path'], num_labels=2)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.config.hidden_size, model.config.hidden_size // 2),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(model.config.hidden_size // 2, 2)
)
model.load_state_dict(torch.load(paths['weight_path']))
train_df = pd.read_csv(
    "./data/misclassified_augment_data/misclassified_augmented_data.tsv",
    sep='\t',
    engine='python',
    quoting=3  
)
origin_df =  pd.read_csv(
    "./data/WNUT-2020-Task-2-Dataset/WNUT-2020-Task-2-Dataset/train.tsv",
    sep='\t',
    engine='python',
    quoting=3  
)

valid_df = pd.read_csv(
    "./data/WNUT-2020-Task-2-Dataset/WNUT-2020-Task-2-Dataset/valid.tsv",
    sep='\t',
    engine='python',
    quoting=3,
    names=["Id", "Text", "Label"]  
)

label_map = {"UNINFORMATIVE": 0, "INFORMATIVE": 1}
train_df["Label"] = train_df["Label"].map(label_map)
origin_df["Label"] = origin_df["Label"].map(label_map)
valid_df["Label"] = valid_df["Label"].map(label_map)

g = torch.Generator()
g.manual_seed(seed)
train_loader = create_mixed_loader(hard_df=train_df, train_df=origin_df, tokenizer=tokenizer, hard_to_normal_ratio=1.0, batch_size=8, shuffle=True, seed=1364)
valid_dataset = TweetDataset(valid_df, tokenizer)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, worker_init_fn=lambda worker_id: set_seed(seed), generator=g)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

MyTrainer = FinetuneTrainer(
    model, 
    train_loader = train_loader, 
    val_loader = valid_loader, 
    tokenizer = tokenizer,
    paths = paths, 
    optimizer_class = optim.AdamW, 
    loss_fn = loss_fn, 
    scheduler_class = ReduceLROnPlateau, 
    device = device,
    model_chosen = model_chosen 
)

MyTrainer.set_hyperparams(lr=1e-6, 
                          num_epochs=2, 
                          save_interval=1,
                          head_lr=5e-6)
MyTrainer.train()
MyTrainer.summarize()