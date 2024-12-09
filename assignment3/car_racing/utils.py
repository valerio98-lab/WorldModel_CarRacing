import torch
import torch.nn as nn
import numpy as np
import os
import re
import glob

from matplotlib import pyplot as plt

torch.manual_seed(42)


def save_model(model, optimizer=None, epoch=None, model_name=None):
    model_name = model_name.split(".")[0]
    torch.save(model.state_dict(), f"{model_name}.pt")

    if os.path.exists("./checkpoints") == False:
        os.makedirs("./checkpoints")

    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, f"./checkpoints/checkpoint_{epoch}.pt")


def load_model(model, optimizer=None, model_name=None, epoch=None, load_checkpoint=False):
    model_name = model_name.split(".")[0]
    if load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{epoch}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.load_state_dict(torch.load(f"{model_name}.pt", weights_only=True))
        print(f"Model loaded from {model_name}.pt")
    return model, optimizer


def _search_files(path):
    return sorted(
        [
            f
            for f in glob.glob(f"{path}_*.pt")
            if re.match(rf"{re.escape(path)}_\d+\.pt$", f)
        ]
    )


def testagent(episode, vae, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Prendi un episodio a caso
    # Numero di immagini da plottare
    num_images_to_plot = np.random.randint(
        0, len(episode.observations), size=5
    )  # Indici casuali

    # Plotta osservazioni e ricostruzioni
    fig, axes = plt.subplots(5, 2, figsize=(10, 15))  # Layout con 5 righe e 2 colonne
    for idx, i in enumerate(
        num_images_to_plot
    ):  # Usa un contatore per accedere agli assi
        # Prendi l'osservazione
        observation = (
            episode.observations[i].unsqueeze(0).to(device)
        )  # Aggiunge dimensione batch

        # Ottieni la ricostruzione
        with torch.no_grad():
            recon, _, _ = vae(observation)

        # Denormalizza le immagini per il plot
        observation_img = observation.squeeze(0).permute(1, 2, 0).cpu().numpy()
        observation_img = observation_img  # Denormalizza

        reconstructed_img = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reconstructed_img = reconstructed_img  # Denormalizza

        # Plot a sinistra: osservazione originale
        axes[idx, 0].imshow(
            observation_img
        )  # Usa il contatore `idx` per accedere agli assi
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title(f"Osservazione {i + 1}")

        # Plot a destra: ricostruzione
        axes[idx, 1].imshow(reconstructed_img)
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title(f"Ricostruzione {i + 1}")
        observation = observation.cpu()

    plt.tight_layout()
    plt.show()
