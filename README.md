# Curso de CLIP e redes de difusão no Colab

Cinco notebooks curtos. Os dois primeiros colocam foto e frase no mesmo espaço com o CLIP. Os dois seguintes geram uma foto a partir de uma frase com o SD-Turbo, e mostram o ruído entrando e saindo. O último junta os dois: gera quatro fotos e deixa o CLIP escolher a que mais combina com o texto.

Cada notebook abre direto no Google Colab, instala o que precisa e traz seis exercícios com correção na hora.

| Aula | Abrir |
|---|---|
| 1. Texto e imagem no mesmo espaço | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/notebooks/01-clip-imagem-e-texto.ipynb) |
| 2. Classificar sem treinar, e buscar a foto | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/notebooks/02-clip-zero-shot-e-busca.ipynb) |
| 3. Uma frase vira uma foto | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/notebooks/03-difusao-texto-para-imagem.ipynb) |
| 4. O ruído entra, e o modelo tira | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/notebooks/04-difusao-ruido-e-passos.ipynb) |
| 5. O CLIP escolhe a foto gerada | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/notebooks/05-clip-escolhe-a-imagem.ipynb) |

No Colab, use **Ambiente de execução → Executar tudo**. As aulas 1 e 2 rodam em CPU. Nas aulas 3 a 5, uma GPU T4 ajuda: o SD-Turbo baixa uns 2 GB na primeira vez e gera cada foto em poucos passos.

## Modelos e fotos

- CLIP `openai/clip-vit-base-patch32`
- SD-Turbo `stabilityai/sd-turbo` (o `guidance_scale` fica em 0; um a quatro passos)
- o ônibus da [Ultralytics](https://www.ultralytics.com/images/bus.jpg), os gatos do [COCO](http://images.cocodataset.org/val2017/000000039769.jpg) e o cachorro do [PyTorch Hub](https://github.com/pytorch/hub/blob/master/images/dog.jpg)

O Turbo erra letra miúda e detalhe de mão. Serve para ver o mecanismo, não para fechar uma arte final.

## Editar

Os notebooks são gerados por `build_notebooks.py`. Os exercícios ficam em `exercicios.py`. Para atualizar, gere, execute e limpe:

```bash
pip install nbformat nbconvert ipykernel torch transformers diffusers accelerate matplotlib pillow
python build_notebooks.py
mkdir -p /tmp/exec && cp notebooks/*.ipynb /tmp/exec/
(cd /tmp/exec && jupyter nbconvert --to notebook --execute --inplace *.ipynb)
cp /tmp/exec/*.ipynb notebooks/
python clean_outputs.py /tmp/exec notebooks/*.ipynb
```

A página do curso fica em [naubergois.com/curso-clip-difusao](https://naubergois.com/curso-clip-difusao/).

Para testar só a correção dos exercícios, gere com as soluções preenchidas:

```bash
SOLUCOES=1 OUT=/tmp/solucoes python build_notebooks.py
```
