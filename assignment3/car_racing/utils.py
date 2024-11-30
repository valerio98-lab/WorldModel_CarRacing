import torch
import torch.nn as nn
import os

def save_model(model, optimizer, scheduler, epoch, model_name):
    torch.save(model.state_dict(), f'{model_name}.pt')

    if os.path.exists('./checkpoints') == False:
        os.makedirs('./checkpoints')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, f'./checkpoints/checkpoint_{epoch}.pt')


def load_model(model, optimizer=None, scheduler=None, model_name=None, epoch=None, load_checkpoint=False):
    if load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{epoch}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        model.load_state_dict(torch.load(f'{model_name}.pt'))
    return model, optimizer, scheduler