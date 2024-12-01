import torch
import torch.nn as nn
import os

torch.manual_seed(42)   

def save_model(model, optimizer=None, epoch=None, model_name=None):
    torch.save(model.state_dict(), f'{model_name}.pt')
    
    if os.path.exists('./checkpoints') == False:
        os.makedirs('./checkpoints')

    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, f'./checkpoints/checkpoint_{epoch}.pt')


def load_model(model, optimizer=None, model_name=None, epoch=None, load_checkpoint=False):
    if load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{epoch}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.load_state_dict(torch.load(f'{model_name}'))
        print(f"Model loaded from {model_name}")
    return model, optimizer