import torch
import torch.nn as nn
import os

from matplotlib import pyplot as plt

torch.manual_seed(42)   

def save_model(model, optimizer=None, epoch=None, model_name=None):
    torch.save(model.state_dict(), f'{model_name}')
    
    if os.path.exists('./checkpoints') == False:
        os.makedirs('./checkpoints')

    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, f'./checkpoints/checkpoint_{epoch}')


def load_model(model, optimizer=None, model_name=None, epoch=None, load_checkpoint=False):
    if load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{epoch}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.load_state_dict(torch.load(f'{model_name}', weights_only=True))
        print(f"Model loaded from {model_name}")
    return model, optimizer




def test_mdn_with_visualization(mdn, vae, latent_dataloader, real_dataloader, device='cuda'):

    mdn.eval()
    vae.eval()

    total_loss = 0
    num_batches = 0

    latent_batch = next(iter(latent_dataloader))
    real_batch = next(iter(real_dataloader))

    latent_obs, actions, _ = latent_batch
    real_obs, _, _ = real_batch

    latent_obs = latent_obs.to(device)
    actions = actions.to(device)
    real_obs = real_obs.to(device)

    idx = 0
    z_t = latent_obs[0, :-1, :] 
    a_t = actions[0, :-1, :]    
    z_t1_target = latent_obs[0, 1:, :]  
    real_frame = real_obs[0]    

    with torch.no_grad():
        alpha, mu, sigma, _ = mdn(z_t.unsqueeze(0), a_t.unsqueeze(0))
        z_t1_pred = mu.squeeze(0) 

    with torch.no_grad():
        real_reconstruction = vae.decoder(z_t1_target).cpu()
        pred_reconstruction = vae.decoder(z_t1_pred).cpu()
    real_episode = real_batch[0]
    real_frame = real_episode[0][0].squeeze(0)
    real_frame_np = real_frame.permute(1, 2, 0).cpu().numpy()
    real_recon_np = real_reconstruction[idx].permute(1, 2, 0).numpy()
    pred_recon_np = pred_reconstruction[idx].permute(1, 2, 0).numpy()

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(real_frame_np)
    axes[0].set_title("Immagine Reale")
    axes[1].imshow(real_recon_np)
    axes[1].set_title("Ricostruzione Reale")
    axes[2].imshow(pred_recon_np)
    axes[2].set_title("Predizione MDN")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Test: calcolo della loss per il batch
    for batch in latent_dataloader:
        latent_obs, actions, _ = batch
        latent_obs = latent_obs.to(device)
        actions = actions.to(device)

        z_t = latent_obs[:, :-1, :]
        z_t1_target = latent_obs[:, 1:, :]
        a_t = actions[:, :-1, :]

        with torch.no_grad():
            alpha, mu, sigma, _ = mdn(z_t, a_t)

            # Loss MDN
            loss = mdn.mdn_loss(alpha, sigma, mu, z_t1_target)
            total_loss += loss.item()
        
