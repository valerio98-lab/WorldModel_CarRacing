import torch
import torch.nn as nn
import numpy as np
import os
import re
import glob
from pathlib import Path

torch.manual_seed(42)


def save_model(model, optimizer=None, epoch=None, model_name=None, checkpoint_path=None):
    model_name = model_name.split(".")[0]
    torch.save(model.state_dict(), f"{model_name}.pt")

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path) == False:
            os.makedirs(checkpoint_path)

    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_{epoch}.pt")


def load_model(
    model,
    optimizer=None,
    model_name=None,
    epoch=None,
    load_checkpoint=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model_name = model_name.split(".")[0]
    if load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{epoch+1}", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.load_state_dict(
            torch.load(f"{model_name}.pt", weights_only=True, map_location=device)
        )
        print(f"Model loaded from {model_name}.pt")
    return model, optimizer


def search_files(path):
    return sorted(
        [
            Path(f)
            for f in glob.glob(f"{path}_*.pt")
            if Path(f).stem.split("_")[0] == Path(path).stem
        ]
    )


def testagent(episode, vae, mdn, device="cuda" if torch.cuda.is_available() else "cpu"):
    import matplotlib.pyplot as plt
    import numpy as np

    num_images_to_plot = np.random.randint(0, len(episode.observations), size=5)

    fig, axes = plt.subplots(5, 3, figsize=(15, 15))
    for idx, i in enumerate(num_images_to_plot):
        observation = episode.observations[i].unsqueeze(0).to(device)

        with torch.no_grad():
            z, _, _ = vae.encoder(observation)
            recon, _, _ = vae(observation)
            alpha, mu, sigma, _ = mdn(z, episode.actions[i].unsqueeze(0).to(device))
            sample = mdn.sample(alpha, mu, sigma)

            mdn_reconstructed_img = (
                vae.decoder(sample).squeeze(0).permute(1, 2, 0).cpu().numpy()
            )

        observation_img = observation.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reconstructed_img = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()

        axes[idx, 0].imshow(observation_img)
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title(f"Osservazione {i + 1}")

        axes[idx, 1].imshow(reconstructed_img)
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title(f"Ricostruzione {i + 1}")

        axes[idx, 2].imshow(mdn_reconstructed_img)
        axes[idx, 2].axis("off")
        axes[idx, 2].set_title(f"Ricostruzione MDN {i + 1}")

    plt.tight_layout()
    plt.show()
