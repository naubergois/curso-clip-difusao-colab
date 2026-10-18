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

## Os 24 exercícios, um notebook cada

Cada exercício também existe como notebook próprio, já executado do começo ao fim: prepara os dados, deixa uma célula para tentar e mostra uma solução conferida.

| Aula | Exercício | Abrir |
|---|---|---|
| 1 | 1. Quem ganhou | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula1-exercicio1-quem-ganhou.ipynb) |
| 1 | 2. A frase, por extenso | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula1-exercicio2-a-frase-por-extenso.ipynb) |
| 1 | 3. O cosseno do vencedor | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula1-exercicio3-o-cosseno-do-vencedor.ipynb) |
| 1 | 4. Acima de 20% | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula1-exercicio4-acima-de-20.ipynb) |
| 1 | 5. Outra disputa | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula1-exercicio5-outra-disputa.ipynb) |
| 1 | 6. O multiplicador | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula1-exercicio6-o-multiplicador.ipynb) |
| 2 | 1. A palavra vencedora | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula2-exercicio1-a-palavra-vencedora.ipynb) |
| 2 | 2. A frase vencedora | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula2-exercicio2-a-frase-vencedora.ipynb) |
| 2 | 3. A forma da frase mudou o vencedor? | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula2-exercicio3-a-forma-da-frase-mudou-o-vencedor.ipynb) |
| 2 | 4. Primeira da busca | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula2-exercicio4-primeira-da-busca.ipynb) |
| 2 | 5. Última da busca | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula2-exercicio5-ultima-da-busca.ipynb) |
| 2 | 6. Folga do primeiro | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula2-exercicio6-folga-do-primeiro.ipynb) |
| 3 | 1. Largura da foto | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula3-exercicio1-largura-da-foto.ipynb) |
| 3 | 2. Canais do latente | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula3-exercicio2-canais-do-latente.ipynb) |
| 3 | 3. Passos que o treino conhece | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula3-exercicio3-passos-que-o-treino-conhece.ipynb) |
| 3 | 4. Outra semente | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula3-exercicio4-outra-semente.ipynb) |
| 3 | 5. Lado do latente | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula3-exercicio5-lado-do-latente.ipynb) |
| 3 | 6. Escala do VAE | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula3-exercicio6-escala-do-vae.ipynb) |
| 4 | 1. Quase sem ruído | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula4-exercicio1-quase-sem-ruido.ipynb) |
| 4 | 2. Quase só ruído | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula4-exercicio2-quase-so-ruido.ipynb) |
| 4 | 3. O tamanho não muda | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula4-exercicio3-o-tamanho-nao-muda.ipynb) |
| 4 | 4. Longe da original | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula4-exercicio4-longe-da-original.ipynb) |
| 4 | 5. Quadros da volta | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula4-exercicio5-quadros-da-volta.ipynb) |
| 4 | 6. Passos do img2img | [Colab](https://colab.research.google.com/github/naubergois/curso-clip-difusao-colab/blob/main/exercicios/aula4-exercicio6-passos-do-img2img.ipynb) |

## Modelos e fotos

- CLIP `openai/clip-vit-base-patch32`
- SD-Turbo `stabilityai/sd-turbo` (o `guidance_scale` fica em 0; um a quatro passos)
- o ônibus da [Ultralytics](https://www.ultralytics.com/images/bus.jpg), os gatos do [COCO](http://images.cocodataset.org/val2017/000000039769.jpg) e o cachorro do [PyTorch Hub](https://github.com/pytorch/hub/blob/master/images/dog.jpg)

O Turbo erra letra miúda e detalhe de mão. Serve para ver o mecanismo, não para fechar uma arte final.

## Editar

Os notebooks das aulas são gerados por `build_notebooks.py`. Os exercícios ficam em `exercicios.py`, e os notebooks avulsos saem de `build_exercicios.py`. Para atualizar, gere, execute e limpe:

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
