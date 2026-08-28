"""Exercício 3.4 — visualiza os filtros da primeira convolução e os mapas
de ativação dos três blocos para uma imagem do conjunto de teste.

Uso: python src/visualizar.py
Gera: outputs/filtros.png e outputs/ativacoes.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG, CLASSES, OUT_DIR
from data import obter_dataloaders
from evaluate import carregar_checkpoint
from utils import obter_dispositivo


def salvar_filtros(modelo, caminho):
    """Os 16 filtros 3x3 da primeira convolução, normalizados para exibição."""
    pesos = modelo.extrator[0][0].weight.data.cpu()
    w = (pesos - pesos.min()) / (pesos.max() - pesos.min())
    cinza = w.shape[1] == 1

    fig, eixos = plt.subplots(2, 8, figsize=(12, 3.2))
    for i, ax in enumerate(eixos.ravel()):
        if cinza:
            ax.imshow(w[i, 0].numpy(), cmap="gray")
        else:
            ax.imshow(w[i].permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(f"f{i}", fontsize=8)
    fig.suptitle(f"Filtros da primeira convolução — {CFG.dataset}")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Filtros salvos em {caminho}")


def salvar_ativacoes(modelo, imagem, rotulo, dispositivo, caminho):
    """Mapas de ativação dos três blocos convolucionais."""
    ativacoes = {}
    for nome, bloco in [("bloco1", modelo.extrator[0]),
                        ("bloco2", modelo.extrator[1]),
                        ("bloco3", modelo.extrator[2])]:
        bloco.register_forward_hook(
            lambda m, entrada, saida, n=nome: ativacoes.__setitem__(n, saida.detach().cpu()))

    with torch.no_grad():
        modelo(imagem.unsqueeze(0).to(dispositivo))

    # desfaz a normalização apenas para exibir a imagem original
    if isinstance(CFG.media, (tuple, list)):
        media = torch.tensor(CFG.media).view(-1, 1, 1)
        desvio = torch.tensor(CFG.desvio).view(-1, 1, 1)
    else:
        media, desvio = CFG.media, CFG.desvio
    original = (imagem * desvio + media).clamp(0, 1)

    fig, eixos = plt.subplots(3, 9, figsize=(13, 4.8))
    for linha, nome in enumerate(["bloco1", "bloco2", "bloco3"]):
        ax = eixos[linha][0]
        if original.shape[0] == 1:
            ax.imshow(original[0].numpy(), cmap="gray")
        else:
            ax.imshow(original.permute(1, 2, 0).numpy())
        ax.axis("off")

        mapas = ativacoes[nome][0]
        for j in range(8):
            eixos[linha][j + 1].imshow(mapas[j].numpy(), cmap="viridis")
            eixos[linha][j + 1].axis("off")
            eixos[linha][j + 1].set_title(f"{nome} c{j}\n{tuple(mapas.shape[1:])}", fontsize=7)

    fig.suptitle(f"Mapas de ativação — {CFG.dataset}: {CLASSES[rotulo]}")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Ativações salvas em {caminho}")


def main():
    dispositivo = obter_dispositivo()
    modelo, _ = carregar_checkpoint(dispositivo)
    modelo.eval()

    teste = obter_dataloaders(CFG)[2]
    x, y = next(iter(teste))

    salvar_filtros(modelo, OUT_DIR / "filtros.png")
    salvar_ativacoes(modelo, x[0], y[0].item(), dispositivo, OUT_DIR / "ativacoes.png")


if __name__ == "__main__":
    main()
