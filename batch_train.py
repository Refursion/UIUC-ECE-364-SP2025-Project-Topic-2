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
from unfreezer import ProgressiveUnfreezer

# Fix random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(364)


class BatchTrainer:
    def __init__(self, model, train_loader, val_loader, tokenizer,paths, optimizer_class, loss_fn, scheduler_class, unfreezer_class, device, model_chosen = 'TinyBERT'):
        """
        Paths is a dictinary with 
        """
        self.model_chosen = model_chosen
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.paths = paths
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.device = device
        self.unfreeser_class = unfreezer_class
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
        self.unfreezer = self.unfreeser_class(self.model, model_chosen=self.model_chosen, mode="epoch", unfreeze_every = self.hyperparams['unfreeze_every'])
        self.unfreezer.freeze_all_except_classifier()
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
            # Call unfreezer
            self.unfreezer.update(epoch=epoch)

            # Early stopping and save best model
            if valid_accuracy > best_valid_acc:
                best_valid_acc = valid_accuracy
                early_stop_counter = 0  
                print(f"New best model found at epoch {epoch+1}, saving...")
                save_name = f"model_epoch_{epoch+1}_valacc_{valid_accuracy:.4f}.pt"
                save_path = os.path.join(self.paths['best_path'], save_name)
                torch.save(self.model.state_dict(), save_path)
                self.tokenizer.save_pretrained(self.paths['best_path'])
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

model_chosen = 'ALBERT' #'TinyBERT'

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

train_df = pd.read_csv(
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
valid_df["Label"] = valid_df["Label"].map(label_map)

g = torch.Generator()
g.manual_seed(364)
train_dataset = TweetDataset(train_df, tokenizer)
valid_dataset = TweetDataset(valid_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, worker_init_fn=lambda worker_id: set_seed(364), generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, worker_init_fn=lambda worker_id: set_seed(364), generator=g)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

MyTrainer = BatchTrainer(
    model, 
    train_loader = train_loader, 
    val_loader = valid_loader, 
    tokenizer = tokenizer,
    paths = paths, 
    optimizer_class = optim.AdamW, 
    loss_fn = loss_fn, 
    scheduler_class = ReduceLROnPlateau, 
    unfreezer_class = ProgressiveUnfreezer, 
    device = device,
    model_chosen = model_chosen 
)

MyTrainer.set_hyperparams(lr=2.5e-5, 
                          num_epochs=20, 
                          save_interval=4,
                          unfreeze_every=4,
                          )
MyTrainer.train()
MyTrainer.summarize()