"""Gera um notebook por exercício em exercicios/*.ipynb.

Cada notebook roda sozinho: instala, refaz o trecho da aula de que o exercício
precisa, mostra o enunciado, deixa uma célula para tentar e executa a solução
com a correção automática.
"""

import os
import re
import unicodedata
from pathlib import Path

import nbformat as nbf

from build_notebooks import CLIP, DIFUSAO, INSTALL_CLIP, INSTALL_DIFUSAO, REPO
from exercicios import CONFERIR, EXERCICIOS

OUT = Path(os.environ.get("OUT", Path(__file__).parent / "exercicios"))

NOMES = {
    "01-clip-imagem-e-texto.ipynb": "Texto e imagem no mesmo espaço",
    "02-clip-zero-shot-e-busca.ipynb": "Classificar sem treinar, e buscar a foto",
    "03-difusao-texto-para-imagem.ipynb": "Uma frase vira uma foto",
    "04-difusao-ruido-e-passos.ipynb": "O ruído entra, e o modelo tira",
    "05-clip-escolhe-a-imagem.ipynb": "O CLIP escolhe a foto gerada",
}

AULAS = {
    "01-clip-imagem-e-texto.ipynb": {
        "install": INSTALL_CLIP,
        "contexto": "A preparação carrega o CLIP e mede a foto do ônibus contra três frases. Ficam prontos `probs`, `cosenos`, `legendas` e `foto`.",
        "setup": [
            CLIP,
            '''import matplotlib.pyplot as plt
from PIL import Image

foto = Image.open(baixar("https://www.ultralytics.com/images/bus.jpg", "onibus.jpg")).convert("RGB")
plt.figure(figsize=(5, 6))
plt.imshow(foto)
plt.axis("off")
plt.show()

legendas = [
    "a photo of a bus",
    "a photo of a cat",
    "a photo of a dog",
]
probs = pontuar(foto, legendas)
for texto, p in zip(legendas, probs):
    print(f"{float(p):6.1%}  {texto}")
barras(legendas, probs, "parte de cada frase")

entradas = processor(text=legendas, images=foto, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    saida = model(**entradas)
img = normalizar(saida.image_embeds)
txt = normalizar(saida.text_embeds)
cosenos = (txt @ img.T).squeeze(1).detach().cpu()
for texto, c in zip(legendas, cosenos):
    print(f"{float(c):6.3f}  {texto}")''',
        ],
    },
    "02-clip-zero-shot-e-busca.ipynb": {
        "install": INSTALL_CLIP,
        "contexto": "A preparação baixa as três fotos, mede a dos gatos com a palavra solta e com a frase, e ordena a busca. Ficam prontos `classes`, `probs_palavra`, `probs_frase`, `fotos` e `notas_busca`.",
        "setup": [
            CLIP,
            '''import matplotlib.pyplot as plt
from PIL import Image

arquivos = {
    "onibus": "https://www.ultralytics.com/images/bus.jpg",
    "gatos": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "cachorro": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
}
fotos = {
    nome: Image.open(baixar(url, f"{nome}.jpg")).convert("RGB")
    for nome, url in arquivos.items()
}

plt.figure(figsize=(10, 3.4))
for i, (nome, img) in enumerate(fotos.items()):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.title(nome)
    plt.axis("off")
plt.tight_layout()
plt.show()

classes = ["cat", "dog", "bus"]
alvo = fotos["gatos"]
probs_palavra = pontuar(alvo, classes)
probs_frase = pontuar(alvo, [f"a photo of a {c}" for c in classes])
print("palavra solta")
for nome, p in zip(classes, probs_palavra):
    print(f"  {float(p):6.1%}  {nome}")
print("frase")
for nome, p in zip(classes, probs_frase):
    print(f"  {float(p):6.1%}  a photo of a {nome}")

consulta = "a photo of a bus on the street"
vetores = vetor_imagem(list(fotos.values()))
notas_busca = (vetores @ vetor_texto([consulta]).T).squeeze(1).detach().cpu()
for nome, nota in sorted(zip(fotos, notas_busca), key=lambda par: float(par[1]), reverse=True):
    print(f"{float(nota):6.3f}  {nome}")''',
        ],
    },
    "03-difusao-texto-para-imagem.ipynb": {
        "install": INSTALL_DIFUSAO,
        "contexto": "A preparação carrega o SD-Turbo e gera o ônibus vermelho na rua de sol, semente 0, dois passos. Ficam prontos `pipe`, `foto_gerada`, `prompt` e a função `gerar`.",
        "setup": [
            DIFUSAO,
            '''prompt = "a red city bus on a sunny street, photo"
foto_gerada = gerar(prompt, semente=0, n_passos=2)
noite = gerar("a red city bus on a snowy street at night, photo", semente=0, n_passos=2)
grade([foto_gerada, noite], ["rua de sol", "rua de neve"])
print("tamanho", foto_gerada.size)
print("canais do latente:", pipe.unet.config.in_channels)
print("passos de treino:", pipe.scheduler.config.num_train_timesteps)
print("lado do latente:", 512 // pipe.vae_scale_factor)
print("escala do VAE:", float(pipe.vae.config.scaling_factor))''',
        ],
    },
    "04-difusao-ruido-e-passos.ipynb": {
        "install": INSTALL_DIFUSAO,
        "contexto": "A preparação estraga a foto do ônibus com ruído, sem rede, e o SD-Turbo tira esse ruído em quatro passos. Ficam prontos `alpha_bar`, `limpa`, `com_ruido` e `passos_modelo`.",
        "setup": [
            '''import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def baixar(url, destino):
    destino = Path(destino)
    if not destino.exists():
        urllib.request.urlretrieve(url, destino)
    return destino


def faixa(imagens, titulos):
    plt.figure(figsize=(3 * len(imagens), 3.2))
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        plt.subplot(1, len(imagens), i + 1)
        if torch.is_tensor(img):
            vis = img.detach().permute(1, 2, 0).clamp(-1, 1)
            vis = ((vis + 1) / 2).numpy()
        else:
            vis = img
        plt.imshow(vis)
        plt.title(titulo, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


foto = Image.open(baixar("https://www.ultralytics.com/images/bus.jpg", "onibus.jpg")).convert("RGB")
arr = torch.from_numpy(np.asarray(foto.resize((128, 128))).copy()).float()
limpa = arr.permute(2, 0, 1) / 255 * 2 - 1

betas = torch.linspace(1e-4, 0.02, 1000)
alpha_bar = torch.cumprod(1 - betas, dim=0)


def com_ruido(x, t, semente=0):
    g = torch.Generator().manual_seed(semente)
    ruido = torch.randn(x.shape, generator=g)
    a = alpha_bar[t]
    return a.sqrt() * x + (1 - a).sqrt() * ruido


tempos = [0, 250, 500, 750, 999]
faixa(
    [com_ruido(limpa, t) for t in tempos],
    [f"t={t}  a={float(alpha_bar[t]):.3f}" for t in tempos],
)
print(f"alpha_bar[0]={float(alpha_bar[0]):.4f}  alpha_bar[999]={float(alpha_bar[999]):.4f}")''',
            DIFUSAO,
            '''passos_modelo = []


def guardar_passo(pipe, passo, tempo, callback_kwargs):
    latentes = callback_kwargs["latents"].to(pipe.vae.dtype)
    with torch.no_grad():
        imagem = pipe.vae.decode(latentes / pipe.vae.config.scaling_factor).sample
    vis = ((imagem[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).detach().cpu().float().numpy()
    passos_modelo.append(vis)
    return callback_kwargs


g = torch.Generator(device="cpu").manual_seed(0)
pipe(
    "a red city bus on a sunny street, photo",
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=g,
    callback_on_step_end=guardar_passo,
    callback_on_step_end_tensor_inputs=["latents"],
)
faixa(passos_modelo, [f"passo {i + 1}" for i in range(len(passos_modelo))])
print("quadros da volta:", len(passos_modelo))

img2img = AutoPipelineForImage2Image.from_pipe(pipe)
origem = foto.resize((512, 512))
prompt_pintura = "a watercolor painting of a red bus on a street"
resultados = [origem]
titulos = ["foto"]
for forca in (0.35, 0.9):
    g = torch.Generator(device="cpu").manual_seed(0)
    resultados.append(img2img(
        prompt_pintura,
        image=origem,
        strength=forca,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=g,
    ).images[0])
    titulos.append(f"strength {forca}")
grade(resultados, titulos)''',
        ],
    },
    "05-clip-escolhe-a-imagem.ipynb": {
        "install": INSTALL_DIFUSAO,
        "contexto": "A preparação gera quatro ônibus, um por semente, e o CLIP dá uma nota para cada um contra a frase. Ficam prontos `notas`, `sementes`, `fotos_geradas`, `vetor_imagem` e `vetor_texto`.",
        "setup": [
            DIFUSAO,
            '''prompt = "a red city bus on a sunny street, photo"
sementes = [0, 1, 2, 3]
fotos_geradas = [gerar(prompt, semente=s, n_passos=2) for s in sementes]
grade(fotos_geradas, [f"semente {s}" for s in sementes])

import gc
del pipe
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()''',
            CLIP,
            '''vetores = vetor_imagem(fotos_geradas)
texto = vetor_texto([prompt])
notas = [float(n) for n in (vetores @ texto.T).squeeze(1).detach().cpu().tolist()]
for semente, nota in zip(sementes, notas):
    print(f"{nota:6.3f}  semente {semente}")
print("venceu a semente", sementes[notas.index(max(notas))])
grade(fotos_geradas, [f"{s}\\n{n:.3f}" for s, n in zip(sementes, notas)])''',
        ],
    },
}


def slug(texto):
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")


def colab_url(caminho):
    return f"https://colab.research.google.com/github/{REPO}/blob/main/{caminho}"


def badge(caminho):
    return (f'<a href="{colab_url(caminho)}" target="_blank">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
            'alt="Abrir no Colab"/></a>')


def novo_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    return nb


def lista_exercicios():
    """24 notebooks: seis exercícios nas aulas 1 a 4. A aula 5 fica no notebook da aula."""
    saida = []
    for aula_nb, aula in EXERCICIOS.items():
        k = int(aula_nb[:2])
        if k > 4:
            continue
        for n, ex in enumerate(aula["lista"], start=1):
            saida.append((aula_nb, k, n, ex, f"aula{k}-exercicio{n}-{slug(ex['titulo'])}.ipynb"))
    return saida


def gerar():
    todos = lista_exercicios()
    OUT.mkdir(exist_ok=True)
    for i, (aula_nb, k, n, ex, arquivo) in enumerate(todos):
        info = AULAS[aula_nb]
        proximo = todos[i + 1] if i + 1 < len(todos) else None
        aula_url = colab_url(f"notebooks/{aula_nb}")
        if proximo:
            fecho = (f"Próximo: [Aula {proximo[1]}, exercício {proximo[2]}: {proximo[3]['titulo']}]"
                     f"({colab_url('exercicios/' + proximo[4])}).")
        else:
            fecho = "Esse foi o último dos 24. O strength decide quanto da foto de partida ainda manda."

        celulas = [
            ("md", f"""{badge('exercicios/' + arquivo)}

# Aula {k}, exercício {n}: {ex['titulo']}

{ex['enunciado'].strip()}

A pergunta usa o que a aula [{k}. {NOMES[aula_nb]}]({aula_url}) deixa pronto. **Ambiente de execução → Executar tudo** carrega isso, deixa uma célula para tentar e, no fim, confere uma solução."""),
            ("code", info["install"]),
        ]
        celulas.append(("md", f"## Preparação\n\n{info['contexto']}"))
        for bloco in info["setup"]:
            celulas.append(("code", bloco))
        celulas += [
            ("form", "# @title Correção automática (rode, não precisa ler)\n"
                     + EXERCICIOS[aula_nb]["preparo"].strip("\n") + "\n" + CONFERIR),
            ("md", f"## Sua vez\n\n{ex['enunciado'].strip()}\n\n*Dica:* {ex['dica']}\n\n"
                   "Troque o `None` pela sua resposta e rode a célula."),
            ("code", f"{ex.get('inicio', '')}resposta = None\n\nconferir({n}, resposta)"),
            ("md", "## Uma solução\n\nCompare com o que você fez. Existem outros caminhos certos: o que vale é a `conferir` aprovar."),
            ("code", f"{ex['solucao'].strip()}\n\nconferir({n}, resposta)"),
            ("md", f"## O que fica\n\n{fecho}"),
        ]

        nb = novo_notebook()
        for tipo, texto in celulas:
            if tipo == "md":
                nb.cells.append(nbf.v4.new_markdown_cell(texto.strip("\n")))
            else:
                cel = nbf.v4.new_code_cell(texto.strip("\n"))
                if tipo == "form":
                    cel.metadata["cellView"] = "form"
                nb.cells.append(cel)
        nbf.write(nb, OUT / arquivo)
    print(f"{len(todos)} notebooks em {OUT}")


if __name__ == "__main__":
    gerar()
