"""Gera os notebooks do curso em notebooks/*.ipynb.

Com SOLUCOES=1, preenche os exercícios com as soluções (para testar a correção).
Com OUT=<pasta>, grava em outra pasta.
"""

import os
from pathlib import Path

import nbformat as nbf

from exercicios import EXERCICIOS, celulas_exercicios

REPO = "naubergois/curso-clip-difusao-colab"
OUT = Path(os.environ.get("OUT", Path(__file__).parent / "notebooks"))
SOLUCOES = os.environ.get("SOLUCOES") == "1"

INSTALL_CLIP = "import sys\n!{sys.executable} -m pip install -q transformers accelerate matplotlib pillow"
INSTALL_DIFUSAO = "import sys\n!{sys.executable} -m pip install -q diffusers transformers accelerate matplotlib pillow"

CLIP = r'''import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def baixar(url, destino):
    destino = Path(destino)
    if not destino.exists():
        urllib.request.urlretrieve(url, destino)
    return destino


def dispositivo():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def como_vetor(saida):
    """A biblioteca devolve o tensor direto, ou embrulhado. A gente quer o tensor."""
    if torch.is_tensor(saida):
        return saida
    for nome in ("pooler_output", "image_embeds", "text_embeds"):
        valor = getattr(saida, nome, None)
        if torch.is_tensor(valor):
            return valor
    raise TypeError(type(saida))


def normalizar(v):
    return v / v.norm(dim=-1, keepdim=True)


device = dispositivo()
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("dispositivo:", device)


def pontuar(imagem, textos):
    """Reparte 100% entre os textos recebidos. Não é uma probabilidade do mundo."""
    entradas = processor(text=textos, images=imagem, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**entradas).logits_per_image[0]
    return logits.softmax(dim=0).detach().cpu()


def vetor_imagem(imagens):
    entradas = processor(images=imagens, return_tensors="pt").to(device)
    with torch.no_grad():
        saida = model.get_image_features(pixel_values=entradas["pixel_values"])
    return normalizar(como_vetor(saida))


def vetor_texto(textos):
    entradas = processor(text=textos, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        saida = model.get_text_features(
            input_ids=entradas["input_ids"],
            attention_mask=entradas["attention_mask"],
        )
    return normalizar(como_vetor(saida))


def barras(rotulos, valores, titulo):
    plt.figure(figsize=(7, 2.6))
    plt.barh(list(rotulos)[::-1], [float(v) for v in valores][::-1])
    plt.xlim(0, 1)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()'''

DIFUSAO = r'''import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")

import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import logging as diffusers_logging
from PIL import Image

diffusers_logging.set_verbosity_error()

MODELO = "stabilityai/sd-turbo"


def baixar(url, destino):
    destino = Path(destino)
    if not destino.exists():
        urllib.request.urlretrieve(url, destino)
    return destino


def dispositivo():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def carregar():
    device, dtype = dispositivo()
    extra = {"variant": "fp16"} if device == "cuda" else {}
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(MODELO, dtype=dtype, **extra)
    except TypeError:
        pipe = AutoPipelineForText2Image.from_pretrained(MODELO, torch_dtype=dtype, **extra)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device == "cpu":
        print("CPU: cada imagem pode levar alguns minutos. No Colab, troque o ambiente para T4.")
    print("dispositivo:", device)
    return pipe


def grade(imagens, titulos):
    n = len(imagens)
    plt.figure(figsize=(3.1 * n, 3.4))
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(titulo, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


pipe = carregar()


def gerar(texto, semente=0, n_passos=2):
    g = torch.Generator(device="cpu").manual_seed(semente)
    return pipe(
        texto,
        num_inference_steps=n_passos,
        guidance_scale=0.0,
        generator=g,
    ).images[0]'''


def badge(nome):
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{nome}"
    return (f'<a href="{url}" target="_blank">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
            'alt="Abrir no Colab"/></a>')


def notebook(nome, celulas):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    if nome in EXERCICIOS:
        celulas = celulas[:-1] + celulas_exercicios(EXERCICIOS[nome], SOLUCOES) + celulas[-1:]
    out = []
    for tipo, texto in celulas:
        texto = texto.strip("\n")
        texto = texto.replace("{BADGE}", badge(nome))
        if tipo == "md":
            cel = nbf.v4.new_markdown_cell(texto)
        else:
            cel = nbf.v4.new_code_cell(texto)
            if tipo == "form":
                cel.metadata["cellView"] = "form"
        out.append(cel)
    nb.cells = out
    OUT.mkdir(exist_ok=True)
    nbf.write(nb, OUT / nome)
    print("ok", nome)


# ---------------------------------------------------------------- 01
notebook("01-clip-imagem-e-texto.ipynb", [
    ("md", """
{BADGE}

# 1. Texto e imagem no mesmo espaço

Uma foto e uma frase não cabem na mesma conta. A foto é um retângulo de pixels. A frase é uma fila de palavras. O CLIP faz as duas virarem vetores, listas de números do mesmo tamanho, e mede se apontam para o mesmo lado.

Se a frase descreve a foto, o ângulo entre os vetores fica pequeno e o produto interno sobe. Se a frase fala de outra coisa, o número cai. O modelo aprendeu isso em centenas de milhões de pares tirados da internet: a legenda que estava junto da foto serviu de rótulo. Ninguém entregou uma lista fechada de classes.

Os pesos são o `clip-vit-base-patch32`. É leve e roda na CPU.

> No Colab: **Ambiente de execução → Executar tudo**. GPU é opcional.
"""),
    ("code", INSTALL_CLIP),
    ("code", CLIP),
    ("md", """
## A foto e três frases

A foto é um ônibus numa rua. As três frases disputam a mesma imagem. Só uma descreve o que está lá.
"""),
    ("code", '''foto = Image.open(baixar("https://www.ultralytics.com/images/bus.jpg", "onibus.jpg")).convert("RGB")
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
assert abs(float(probs.sum()) - 1) < 1e-4'''),
    ("md", """
## O cosseno, antes de virar porcentagem

O softmax reparte 100% entre as frases que você escreveu. Se as três forem ruins, a "vencedora" ainda leva a maior fatia. O número diz quem ganhou a disputa, não que a foto seja aquilo no mundo.

Por baixo, a conta é um cosseno: 1 é a mesma direção, 0 é um ângulo reto. O modelo multiplica esse cosseno por um número aprendido (`logit_scale`) antes do softmax, e a disputa fica bem mais decidida do que o cosseno cru sugere. Um cosseno perto de 0,3 pode virar quase 100%. Gato e cachorro somem na porcentagem. No cosseno, eles ainda estão lá, só que mais longe.
"""),
    ("code", '''entradas = processor(text=legendas, images=foto, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    saida = model(**entradas)
img = normalizar(saida.image_embeds)
txt = normalizar(saida.text_embeds)
cosenos = (txt @ img.T).squeeze(1).detach().cpu()
escala = float(model.logit_scale.exp().detach())
print(f"multiplicador do cosseno: {escala:.1f}")
for texto, c in zip(legendas, cosenos):
    print(f"{float(c):6.3f}  {texto}")
assert int(probs.argmax()) == int(cosenos.argmax())'''),
    ("md", """
## O que fica

CLIP não desenha caixa e não gera imagem. Ele responde qual texto combina com a foto, ou qual foto combina com o texto. A porcentagem só vale dentro das opções que você entregou. Quem desenha a partir da frase é a rede de difusão.
"""),
])

# ---------------------------------------------------------------- 02
notebook("02-clip-zero-shot-e-busca.ipynb", [
    ("md", """
{BADGE}

# 2. Classificar sem treinar, e buscar a foto

Classificar, no jeito antigo, pede fotos já separadas nas classes que você escolheu antes. O CLIP inverte a ordem. Você escreve os nomes, ele compara cada nome com a foto e aponta o que chegou mais perto. O nome disso é classificação *zero-shot*: zero exemplos novos, só o texto.

O mesmo vetor serve para busca. A frase vira a consulta. As fotos entram numa fila, da mais parecida para a menos.

> No Colab: **Ambiente de execução → Executar tudo**. GPU é opcional.
"""),
    ("code", INSTALL_CLIP),
    ("code", CLIP),
    ("code", '''arquivos = {
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
plt.show()'''),
    ("md", """
## A forma da frase muda a nota

O CLIP leu legendas, não etiquetas de planilha. `cat` e `a photo of a cat` não caem no mesmo ponto. Compare as duas disputas na foto dos gatos.
"""),
    ("code", '''classes = ["cat", "dog", "bus"]
alvo = fotos["gatos"]
probs_palavra = pontuar(alvo, classes)
probs_frase = pontuar(alvo, [f"a photo of a {c}" for c in classes])

print("palavra solta")
for nome, p in zip(classes, probs_palavra):
    print(f"  {float(p):6.1%}  {nome}")
print("frase")
for nome, p in zip(classes, probs_frase):
    print(f"  {float(p):6.1%}  a photo of a {nome}")

barras(classes, probs_palavra, "só a palavra")
barras([f"a photo of a {c}" for c in classes], probs_frase, "frase completa")'''),
    ("md", """
## Singular, plural, outro bicho

A foto tem dois gatos. A frase `a photo of two cats` chega mais perto do que o singular, e o cachorro fica de fora. A frase mais específica, quando é verdadeira, sobe. Isso ainda não é um contador: a nota mede o texto que você escreveu. Troque a frase, ou esconda um dos gatos, e a ordem muda.
"""),
    ("code", '''opcoes = ["a photo of a cat", "a photo of two cats", "a photo of a dog"]
probs_contagem = pontuar(alvo, opcoes)
for texto, p in zip(opcoes, probs_contagem):
    print(f"{float(p):6.1%}  {texto}")
barras(opcoes, probs_contagem, "singular, plural, outro")'''),
    ("md", """
## Busca: uma frase, três fotos

A consulta é uma frase. Cada foto já virou vetor. O cosseno ordena a fila. Não há softmax aqui, porque a pergunta não é "reparta 100%". É "quem chega mais perto".
"""),
    ("code", '''consulta = "a photo of a bus on the street"
vetores = vetor_imagem(list(fotos.values()))
notas_busca = (vetores @ vetor_texto([consulta]).T).squeeze(1).detach().cpu()
for nome, nota in sorted(zip(fotos, notas_busca), key=lambda par: float(par[1]), reverse=True):
    print(f"{float(nota):6.3f}  {nome}")
assert list(fotos)[int(notas_busca.argmax())] == "onibus"'''),
    ("md", """
## O que fica

Zero-shot é uma disputa de cossenos com textos escritos na hora. Na foto dos gatos, a palavra solta e a frase `a photo of a ...` elegeram a mesma classe. O que mudou foi a folga. O plural verdadeiro levou a disputa seguinte, e a busca devolveu o ônibus. A nota segue a frase. Ela não faz um censo, e uma frase genérica ainda ganha quando a específica está mal escrita.
"""),
])

# ---------------------------------------------------------------- 03
notebook("03-difusao-texto-para-imagem.ipynb", [
    ("md", """
{BADGE}

# 3. Uma frase vira uma foto

Gerar uma foto a partir de uma frase é o caminho inverso de comparar. A rede de difusão começa no chiado, aquele ruído de televisão fora do ar, e a cada passo apaga um pouco. A frase empurra o chiado na direção do que ela descreve. No último passo, o chiado virou imagem.

O modelo é o SD-Turbo (`stabilityai/sd-turbo`). Ele foi destilado para fazer o trajeto em um a quatro passos, em vez de cinquenta. Por isso o `guidance_scale` fica em zero: o peso da frase já foi assado no treino. Letra pequena e mão de gente ainda saem tortas. Dá para ver o mecanismo. Não substitui um estúdio.

A semente (`seed`) só fixa o acaso. A mesma frase, com a mesma semente, cai na mesma foto.

> GPU faz diferença grande: **Ambiente de execução → Alterar o tipo de ambiente → T4 GPU**.
"""),
    ("code", INSTALL_DIFUSAO),
    ("code", DIFUSAO),
    ("md", """
## A primeira foto

Dois passos. Semente zero. A frase pede um ônibus vermelho numa rua de sol.
"""),
    ("code", '''passos = 2
guia = 0.0
prompt = "a red city bus on a sunny street, photo"
foto_gerada = gerar(prompt, semente=0, n_passos=passos)
print("guidance_scale", guia, "| passos", passos, "| tamanho", foto_gerada.size)
plt.figure(figsize=(5, 5))
plt.imshow(foto_gerada)
plt.axis("off")
plt.show()
assert foto_gerada.size == (512, 512)'''),
    ("md", """
## A frase manda no assunto

Mesma semente, outro tempo. Se a semente fica e a frase muda, muda o assunto, não o "sorteio" do ruído.
"""),
    ("code", '''noite = gerar("a red city bus on a snowy street at night, photo", semente=0, n_passos=passos)
grade(
    [foto_gerada, noite],
    ["rua de sol", "rua de neve"],
)'''),
    ("md", """
## O que a rede pinta não é o pixel

Por dentro, a rede não mexe nos 512×512×3 números da foto. Ela mexe num latente: uma versão comprimida, com 4 canais e lado 512 dividido por 8. Um decodificador (o VAE) abre isso em pixels no fim. Por isso `in_channels` do UNet é 4, e o agendador ainda conhece os 1000 passos do treino, mesmo quando a gente só anda dois.
"""),
    ("code", '''print("canais do latente:", pipe.unet.config.in_channels)
print("fator do VAE:", pipe.vae_scale_factor)
print("lado do latente:", 512 // pipe.vae_scale_factor)
print("passos de treino:", pipe.scheduler.config.num_train_timesteps)
print("escala do VAE:", float(pipe.vae.config.scaling_factor))'''),
    ("md", """
## O que fica

Uma frase, uma semente e poucos passos já devolvem uma foto. O Turbo não usa o botão de volume da frase (`guidance_scale` fica em 0). Quem quiser ver o ruído entrar e sair, passo a passo, olha a ida sem rede e a volta com ela.
"""),
])

# ---------------------------------------------------------------- 04
notebook("04-difusao-ruido-e-passos.ipynb", [
    ("md", """
{BADGE}

# 4. O ruído entra, e o modelo tira

O treino de uma rede de difusão não começa pela foto pronta. Começa pela foto estragada de propósito. Soma-se ruído gaussiano, um pouco ou muito, e a rede aprende a apontar qual ruído foi somado. Na hora de gerar, não existe foto. Existe ruído, e a rede vai tirando o que ela acredita que não deveria estar ali.

Dá para ver as duas metades. Primeiro a ida, sem rede nenhuma: a mesma foto com cada vez mais ruído. Depois a volta, com o SD-Turbo, passo a passo. No fim a foto de partida volta por outro caminho, o img2img, e o `strength` decide quanto dela sobrevive.

> A ida roda na CPU. A volta pede GPU: **Ambiente de execução → Alterar o tipo de ambiente → T4 GPU**.
"""),
    ("code", INSTALL_DIFUSAO),
    ("code", '''import urllib.request
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
print(f"shape: {tuple(limpa.shape)}")
print(f"alpha_bar[0]={float(alpha_bar[0]):.4f}  alpha_bar[999]={float(alpha_bar[999]):.4f}")'''),
    ("md", """
## A volta: quatro passos saindo do chiado

A faixa acima é pixel. O SD-Turbo trabalha no latente, a versão comprimida da imagem, e um decodificador abre cada passo para a gente olhar. Os primeiros saem lavados de propósito: ainda tem ruído demais para virar foto. O `guidance_scale` continua em zero.
"""),
    ("code", DIFUSAO),
    ("code", '''passos_modelo = []


def guardar_passo(pipe, passo, tempo, callback_kwargs):
    latentes = callback_kwargs["latents"].to(pipe.vae.dtype)
    with torch.no_grad():
        imagem = pipe.vae.decode(latentes / pipe.vae.config.scaling_factor).sample
    vis = ((imagem[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).detach().cpu().float().numpy()
    passos_modelo.append(vis)
    return callback_kwargs


g = torch.Generator(device="cpu").manual_seed(0)
foto_volta = pipe(
    "a red city bus on a sunny street, photo",
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=g,
    callback_on_step_end=guardar_passo,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]

faixa(passos_modelo, [f"passo {i + 1}" for i in range(len(passos_modelo))])
plt.figure(figsize=(4, 4))
plt.imshow(foto_volta)
plt.axis("off")
plt.title("foto final")
plt.show()
assert len(passos_modelo) == 4'''),
    ("md", """
## Img2img: quanto da foto original sobrevive

Aqui a rede não parte do chiado puro. Ela parte da foto, com ruído misturado na medida do `strength`. Perto de 0, a foto quase não muda. Perto de 1, a frase recomeça. Com 4 passos de inferência, o img2img anda `int(4 * strength)` passos. 0,35 vira 1 passo. 0,9 vira 3.
"""),
    ("code", '''img2img = AutoPipelineForImage2Image.from_pipe(pipe)
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
grade(resultados, titulos)'''),
    ("md", """
## O que fica

A ida é uma conta: `alpha_bar` pesa a foto, o resto é ruído, e no fim do treino a rede só aprende a apontar esse ruído. A volta que a gente viu é curta de propósito, quatro passos, num latente. O `strength` é o botão de quanto a foto de partida ainda manda. Isso ainda não é o treino da rede. É o processo com os pesos que alguém já treinou.
"""),
])

# ---------------------------------------------------------------- 05
notebook("05-clip-escolhe-a-imagem.ipynb", [
    ("md", """
{BADGE}

# 5. O CLIP escolhe a foto gerada

Os dois modelos fecham um ciclo. A difusão desenha várias fotos para a mesma frase, porque cada semente de ruído cai num lugar diferente. O CLIP, que não desenha, olha cada foto e mede a distância até a frase. A mais perto ganha.

O número não é gosto. É semelhança com a legenda. Uma foto borrada do assunto certo pode ganhar de uma foto nítida do assunto errado.

> GPU: **Ambiente de execução → Alterar o tipo de ambiente → T4 GPU**.
"""),
    ("code", INSTALL_DIFUSAO),
    ("code", DIFUSAO),
    ("md", """
## Quatro sementes, a mesma frase
"""),
    ("code", '''prompt = "a red city bus on a sunny street, photo"
sementes = [0, 1, 2, 3]
fotos_geradas = [gerar(prompt, semente=s, n_passos=2) for s in sementes]
grade(fotos_geradas, [f"semente {s}" for s in sementes])
assert len(fotos_geradas) == 4'''),
    ("md", """
## Soltar o gerador e chamar o juiz

As fotos já estão na memória. O Turbo ocupa a GPU à toa daqui para a frente. O CLIP é leve e só precisa das quatro imagens e da frase.
"""),
    ("code", '''import gc

del pipe
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()'''),
    ("code", CLIP),
    ("code", '''vetores = vetor_imagem(fotos_geradas)
texto = vetor_texto([prompt])
notas = (vetores @ texto.T).squeeze(1).detach().cpu().tolist()
notas = [float(n) for n in notas]
for semente, nota in zip(sementes, notas):
    print(f"{nota:6.3f}  semente {semente}")
vencedora = notas.index(max(notas))
print("venceu a semente", sementes[vencedora])
grade(fotos_geradas, [f"{s}\\n{n:.3f}" for s, n in zip(sementes, notas)])'''),
    ("md", """
## O que fica

Difusão propõe. CLIP ordena. A frase serve de pedido para um e de régua para o outro. A régua mede a legenda, não se a foto está bonita, e o Turbo ainda erra letra e detalhe miúdo. Com as duas peças dá para gerar um punhado e ficar com a que mais combina com o texto.
"""),
])
