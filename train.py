import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

MANUAL_SEED = 42
torch.manual_seed(42)
torch.cuda.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed_all(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _init_fn(worker_id):
    np.random.seed(42)
                   
def train(train_dataset, val_dataset, model, criterion, save_title: str, n_epochs: int=10, eval_period: int=2):
    
    train_losses, val_losses, val_loss = [], [], None
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, worker_init_fn=_init_fn, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, worker_init_fn=_init_fn, pin_memory=True)
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    
    for n_epoch in range(1, n_epochs + 1):
        model.train()
        cum_loss = 0
        
        with tqdm(train_dataloader, unit='batch') as tepoch:
                      
            for item in tepoch:
                tepoch.set_description(f"Epoch {n_epoch}")
                
                hdmap = item['feature_maps'].to(device)
                agent_state_vector = item['agent_state_vector'].to(device)
                ground_truth = item['ground_truth_trajectory'].to(device)
                
                optimizer.zero_grad()
                preds = model(hdmap, agent_state_vector)
                loss = criterion(preds, ground_truth)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                optimizer.step()
                
                current_loss = loss.item()
                cum_loss += current_loss
                
                tepoch.set_postfix(loss=current_loss, validation_loss=val_loss if val_loss is not None else '?')
        
        train_losses.append(cum_loss / len(train_dataloader))
        
        if n_epoch % eval_period == 0:
            val_loss = evaluate(model, val_dataloader, criterion)
            val_losses.append(val_loss)
    
    save_title += f'_seed_{MANUAL_SEED}.pt'
    torch.save(model, save_title)
    
    print('model is successfully saved!')
    
    return train_losses, val_losses

def evaluate(model, val_dataloader, criterion):
    model.eval()
    cum_loss = 0
    
    with torch.no_grad():
        with tqdm(val_dataloader, unit='batch') as tepoch:
            tepoch.set_description("Validation")
            for item in tepoch:
                hdmap = item['feature_maps'].to(device)
                agent_state_vector = item['agent_state_vector'].to(device)
                ground_truth = item['ground_truth_trajectory'].to(device)

                output = model(hdmap, agent_state_vector)
                loss = criterion(output, ground_truth)
                cum_loss += loss.item()
    
    return cum_loss / len(val_dataloader)
