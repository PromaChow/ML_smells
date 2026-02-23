import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class LungDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.dataframe.items()}
        input_ids = self.tokenizer.encode(item['text'], 
                                            add_special_tokens=True, 
                                            max_length=self.max_len, 
                                            padding="max_length", 
                                            truncation=True)
        attention_mask = torch.tensor(input_ids['attention_mask'])
        item.update({'input_ids': input_ids['input_ids'], 'attention_mask': attention_mask})
        return item
    
    def __len__(self):
        return len(self.dataframe)

# Hyperparameters
max_len = 512
batch_size = 32
num_epochs = 5
learning_rate = 1e-06

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize data and split into training and validation sets
train_dataset = LungDataset(train_df, tokenizer, max_len)
val_dataset = LungDataset(test_df, tokenizer, max_len)

# Initialize model
model = BertModel.from_pretrained('bert-base-uncased')

# Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Train model
for epoch in range(num_epochs):
    model.train()
    for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['pressure'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Evaluate on validation set
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in DataLoader(val_dataset, batch_size=batch_size, shuffle=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['pressure'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            eval_loss += loss.item()
    
    print('Epoch {}: Loss: {:.4f}'.format(epoch+1, eval_loss/len(val_dataset)))

# Make predictions on test set
test_preds = []
for batch in DataLoader(test_df, batch_size=batch_size, shuffle=False):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    
    test_preds.append(predictions.cpu().numpy())
test_preds = np.concatenate(test_preds)

# Calculate metrics
val_mae = mean_absolute_error(test_df['pressure'].to_list(), test_preds)
print('Validation MAE: {:.4f}'.format(val_mae))