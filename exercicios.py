"""Exercícios com correção automática, seis por aula.

Cada exercício tem enunciado, dica, solução e uma verificação que calcula a
resposta esperada a partir dos dados do próprio notebook.
"""

CONFERIR = '''
_acertos = {}


def conferir(n, resposta):
    if resposta is None:
        print(f"Exercício {n}: ainda sem resposta.")
        return
    try:
        certo = bool(VERIFICAR[n](resposta))
    except Exception as erro:
        certo = False
        print(f"  (a verificação deu erro: {erro})")
    _acertos[n] = certo
    print(f"Exercício {n}: " + ("✅ certo!" if certo else "❌ ainda não. Releia o enunciado e tente de novo."))


def placar():
    print(f"{sum(_acertos.values())} de {len(VERIFICAR)} exercícios certos.")
'''


def celulas_exercicios(aula, solucoes=False):
    celulas = [
        ("md", """
## Exercícios

Seis perguntas sobre o que acabou de rodar. Escreva a resposta na variável `resposta` e rode a célula. `conferir` diz na hora se fechou. Se travar, abra **Ver solução**.

Rode antes a célula de preparação. Ela só carrega as verificações.
"""),
        ("form", "# @title Preparação dos exercícios (rode, não precisa ler)\n" + aula["preparo"].strip("\n") + "\n" + CONFERIR),
    ]
    for n, ex in enumerate(aula["lista"], start=1):
        solucao = ex["solucao"].strip("\n")
        celulas.append(("md", f"""
### Exercício {n}. {ex["titulo"]}

{ex["enunciado"].strip()}

<details>
<summary>Ver solução</summary>

```python
{solucao}
```

</details>
"""))
        corpo = solucao if solucoes else f"# Dica: {ex['dica']}\n{ex.get('inicio', '')}resposta = None"
        celulas.append(("code", f"{corpo}\n\nconferir({n}, resposta)"))
    celulas.append(("code", "placar()"))
    return celulas


AULA1 = {
    "preparo": '''
_probs = [float(p) for p in probs]
_cos = [float(c) for c in cosenos]
_frases_extra = [
    "a photo of a yellow bus",
    "a photo of a cat",
    "a photo of a dog",
]
_extra = [float(p) for p in pontuar(foto, _frases_extra)]
indice_vencedor = _probs.index(max(_probs))

VERIFICAR = {
    1: lambda r, i=indice_vencedor: int(r) == i,
    2: lambda r, i=indice_vencedor: r == legendas[i],
    3: lambda r, i=indice_vencedor: abs(float(r) - _cos[i]) < 1e-3,
    4: lambda r: int(r) == sum(p > 0.2 for p in _probs),
    5: lambda r: r == _frases_extra[_extra.index(max(_extra))],
    6: lambda r: abs(float(r) - float(model.logit_scale.exp().detach())) < 1e-2,
}
''',
    "lista": [
        {
            "titulo": "Quem ganhou",
            "enunciado": "Qual é o índice da frase vencedora em `legendas`? O índice começa em 0.",
            "dica": "probs guarda uma nota por frase, na mesma ordem.",
            "solucao": "resposta = int(probs.argmax())",
        },
        {
            "titulo": "A frase, por extenso",
            "enunciado": "Escreva o texto da frase vencedora, não o índice.",
            "dica": "Use o índice para entrar em legendas.",
            "solucao": "resposta = legendas[int(probs.argmax())]",
        },
        {
            "titulo": "O cosseno do vencedor",
            "enunciado": "Qual é o cosseno (entre -1 e 1) da frase que ganhou? Esse número está em `cosenos`, antes do softmax.",
            "dica": "O índice do maior valor de probs serve para os dois.",
            "solucao": "resposta = float(cosenos[int(probs.argmax())])",
        },
        {
            "titulo": "Acima de 20%",
            "enunciado": "Quantas frases ficaram com probabilidade maior que 0,2?",
            "dica": "probs é um tensor. Dá para somar uma comparação.",
            "solucao": "resposta = int((probs > 0.2).sum())",
        },
        {
            "titulo": "Outra disputa",
            "enunciado": "Rode `pontuar` de novo na mesma foto, agora com estas três frases: `a photo of a yellow bus`, `a photo of a cat`, `a photo of a dog`. Responda com o texto da vencedora.",
            "dica": "pontuar devolve um tensor na mesma ordem das frases.",
            "solucao": '''frases = ["a photo of a yellow bus", "a photo of a cat", "a photo of a dog"]
notas = pontuar(foto, frases)
resposta = frases[int(notas.argmax())]''',
        },
        {
            "titulo": "O multiplicador",
            "enunciado": "Qual é o número pelo qual o modelo multiplica o cosseno antes do softmax? É `model.logit_scale.exp()`.",
            "dica": "Converta para float.",
            "solucao": "resposta = float(model.logit_scale.exp().detach())",
        },
    ],
}

AULA2 = {
    "preparo": '''
_classes = list(classes)
_palavra = [float(p) for p in probs_palavra]
_frase = [float(p) for p in probs_frase]
_nomes = list(fotos.keys())
_busca = [float(n) for n in notas_busca]
_ordem = [nome for _, nome in sorted(zip(_busca, _nomes), reverse=True)]
_i_palavra = _palavra.index(max(_palavra))
_i_frase = _frase.index(max(_frase))

VERIFICAR = {
    1: lambda r: r == _classes[_i_palavra],
    2: lambda r: r == _classes[_i_frase],
    3: lambda r: bool(r) == (_i_palavra != _i_frase),
    4: lambda r: r == _ordem[0],
    5: lambda r: r == _ordem[-1],
    6: lambda r: abs(float(r) - (_busca[_nomes.index(_ordem[0])] - _busca[_nomes.index(_ordem[1])])) < 1e-3,
}
''',
    "lista": [
        {
            "titulo": "A palavra vencedora",
            "enunciado": "Na foto dos gatos, qual classe ganhou quando os textos eram só as palavras (`cat`, `dog`, `bus`)? Responda com a palavra.",
            "dica": "probs_palavra está na mesma ordem de classes.",
            "solucao": "resposta = classes[int(probs_palavra.argmax())]",
        },
        {
            "titulo": "A frase vencedora",
            "enunciado": "E quando o texto virou `a photo of a ...`? Responda de novo com a classe (`cat`, `dog` ou `bus`), não com a frase inteira.",
            "dica": "probs_frase usa a mesma ordem de classes.",
            "solucao": "resposta = classes[int(probs_frase.argmax())]",
        },
        {
            "titulo": "A forma da frase mudou o vencedor?",
            "enunciado": "O índice vencedor mudou entre a palavra solta e a frase `a photo of a ...`? Responda `True` ou `False`.",
            "dica": "Compare argmax de probs_palavra com o de probs_frase.",
            "solucao": "resposta = int(probs_palavra.argmax()) != int(probs_frase.argmax())",
        },
        {
            "titulo": "Primeira da busca",
            "enunciado": "Na busca pela frase, qual foto ficou em primeiro? Responda com a chave do dicionário `fotos` (`onibus`, `gatos` ou `cachorro`).",
            "dica": "notas_busca segue a ordem de fotos.keys().",
            "solucao": "resposta = list(fotos.keys())[int(notas_busca.argmax())]",
        },
        {
            "titulo": "Última da busca",
            "enunciado": "Qual foto ficou em último na mesma busca?",
            "dica": "argmin é o inverso de argmax.",
            "solucao": "resposta = list(fotos.keys())[int(notas_busca.argmin())]",
        },
        {
            "titulo": "Folga do primeiro",
            "enunciado": "Qual a diferença de cosseno entre a primeira foto e a segunda na busca?",
            "dica": "Ordene as notas e subtraia.",
            "solucao": '''ordem = sorted(float(n) for n in notas_busca)
resposta = ordem[-1] - ordem[-2]''',
        },
    ],
}

AULA3 = {
    "preparo": '''
from PIL import Image


def _pixels(img):
    return list(img.resize((16, 16)).getdata())


VERIFICAR = {
    1: lambda r: int(r) == foto_gerada.size[0],
    2: lambda r: int(r) == pipe.unet.config.in_channels,
    3: lambda r: int(r) == pipe.scheduler.config.num_train_timesteps,
    4: lambda r: (
        isinstance(r, Image.Image)
        and r.size == (512, 512)
        and _pixels(r) != _pixels(foto_gerada)
    ),
    5: lambda r: int(r) == 512 // pipe.vae_scale_factor,
    6: lambda r: abs(float(r) - float(pipe.vae.config.scaling_factor)) < 1e-4,
}
''',
    "lista": [
        {
            "titulo": "Largura da foto",
            "enunciado": "Quantos pixels de largura tem `foto_gerada`?",
            "dica": "Uma imagem PIL tem .size, que é (largura, altura).",
            "solucao": "resposta = foto_gerada.size[0]",
        },
        {
            "titulo": "Canais do latente",
            "enunciado": "O UNet não recebe RGB. Quantos canais tem o latente que entra nele?",
            "dica": "pipe.unet.config.in_channels.",
            "solucao": "resposta = pipe.unet.config.in_channels",
        },
        {
            "titulo": "Passos que o treino conhece",
            "enunciado": "O Turbo anda poucos passos na geração, mas o agendador ainda conhece a escala inteira do treino. Quantos passos de treino estão em `pipe.scheduler.config.num_train_timesteps`?",
            "dica": "É um inteiro, na casa dos mil.",
            "solucao": "resposta = pipe.scheduler.config.num_train_timesteps",
        },
        {
            "titulo": "Outra semente",
            "enunciado": "Gere de novo o mesmo prompt, com semente 4 e 1 passo. A resposta é a imagem. Ela precisa ter 512×512 e não pode ser a mesma `foto_gerada`.",
            "dica": "gerar(prompt, semente=4, n_passos=1) devolve uma imagem PIL.",
            "solucao": "resposta = gerar(prompt, semente=4, n_passos=1)",
        },
        {
            "titulo": "Lado do latente",
            "enunciado": "A rede pinta um quadrado menor, de lado `512 / pipe.vae_scale_factor`. Qual é esse lado, em pixels?",
            "dica": "Divisão inteira.",
            "solucao": "resposta = 512 // pipe.vae_scale_factor",
        },
        {
            "titulo": "Escala do VAE",
            "enunciado": "Antes de decodificar, o latente é dividido por `pipe.vae.config.scaling_factor`. Qual é esse número?",
            "dica": "Converta para float.",
            "solucao": "resposta = float(pipe.vae.config.scaling_factor)",
        },
    ],
}

AULA4 = {
    "preparo": '''
_a0 = float(alpha_bar[0])
_a999 = float(alpha_bar[999])
_e0 = float((com_ruido(limpa, 0) - limpa).abs().mean())
_e999 = float((com_ruido(limpa, 999) - limpa).abs().mean())

VERIFICAR = {
    1: lambda r: abs(float(r) - _a0) < 1e-5,
    2: lambda r: abs(float(r) - _a999) < 1e-5,
    3: lambda r: tuple(int(x) for x in r) == tuple(limpa.shape),
    4: lambda r: bool(r) is True and _e999 > _e0,
    5: lambda r: int(r) == len(passos_modelo),
    6: lambda r: tuple(int(x) for x in r) == (int(4 * 0.35), int(4 * 0.9)),
}
''',
    "lista": [
        {
            "titulo": "Quase sem ruído",
            "enunciado": "Qual é o `alpha_bar` no passo 0? Perto de 1, a foto original ainda pesa quase tudo.",
            "dica": "alpha_bar[0].",
            "solucao": "resposta = float(alpha_bar[0])",
        },
        {
            "titulo": "Quase só ruído",
            "enunciado": "Qual é o `alpha_bar` no passo 999?",
            "dica": "O último índice de uma escala de 1000 passos é 999.",
            "solucao": "resposta = float(alpha_bar[999])",
        },
        {
            "titulo": "O tamanho não muda",
            "enunciado": "Qual é o `shape` do tensor devolvido por `com_ruido(limpa, 500)`? Responda com uma tupla.",
            "dica": "Somar ruído não muda altura, largura nem canais.",
            "solucao": "resposta = tuple(com_ruido(limpa, 500).shape)",
        },
        {
            "titulo": "Longe da original",
            "enunciado": "A foto no passo 999 fica mais longe da original do que a foto no passo 0? Responda `True` ou `False`. Distância aqui é a média do valor absoluto da diferença.",
            "dica": "(com_ruido(limpa, t) - limpa).abs().mean().",
            "solucao": '''erro0 = (com_ruido(limpa, 0) - limpa).abs().mean()
erro999 = (com_ruido(limpa, 999) - limpa).abs().mean()
resposta = bool(erro999 > erro0)''',
        },
        {
            "titulo": "Quadros da volta",
            "enunciado": "Quantos quadros a volta do Turbo guardou em `passos_modelo`?",
            "dica": "É a quantidade de passos de inferência.",
            "solucao": "resposta = len(passos_modelo)",
        },
        {
            "titulo": "Passos do img2img",
            "enunciado": "O img2img anda `int(passos * strength)` passos. Com 4 passos de inferência, quantos rodam com strength 0,35 e com strength 0,9? Responda a tupla, nessa ordem.",
            "dica": "int() corta a parte decimal.",
            "solucao": "resposta = (int(4 * 0.35), int(4 * 0.9))",
        },
    ],
}

AULA5 = {
    "preparo": '''
_notas = [float(n) for n in notas]
_melhor = _notas.index(max(_notas))
_pior = _notas.index(min(_notas))
_media = sum(_notas) / len(_notas)
_vet = vetor_imagem(fotos_geradas)
_gato = vetor_texto(["a photo of a cat"])
_i_gato = int((_vet @ _gato.T).squeeze().argmax())

VERIFICAR = {
    1: lambda r: int(r) == _melhor,
    2: lambda r: int(r) == sementes[_melhor],
    3: lambda r: int(r) == _pior,
    4: lambda r: abs(float(r) - (_notas[_melhor] - _notas[_pior])) < 1e-3,
    5: lambda r: int(r) == _i_gato,
    6: lambda r: int(r) == sum(n > _media for n in _notas),
}
''',
    "lista": [
        {
            "titulo": "A mais perto da frase",
            "enunciado": "Qual é o índice da foto com a maior nota do CLIP? Se empatar, vale o primeiro.",
            "dica": "notas está na mesma ordem de fotos_geradas.",
            "solucao": "resposta = notas.index(max(notas))",
        },
        {
            "titulo": "A semente vencedora",
            "enunciado": "Qual semente gerou essa foto?",
            "dica": "sementes segue a mesma ordem.",
            "solucao": "resposta = sementes[notas.index(max(notas))]",
        },
        {
            "titulo": "A mais longe",
            "enunciado": "Qual é o índice da pior nota?",
            "dica": "min(notas) e list.index.",
            "solucao": "resposta = notas.index(min(notas))",
        },
        {
            "titulo": "Folga",
            "enunciado": "Qual a diferença entre a maior nota e a menor?",
            "dica": "Subtração direta.",
            "solucao": "resposta = max(notas) - min(notas)",
        },
        {
            "titulo": "Outra frase, as mesmas fotos",
            "enunciado": "Sem gerar de novo, meça as quatro fotos contra a frase `a photo of a cat`. Qual índice fica mais perto dela?",
            "dica": "vetor_imagem(fotos_geradas) @ vetor_texto([...]).T",
            "solucao": '''vet = vetor_imagem(fotos_geradas)
texto = vetor_texto(["a photo of a cat"])
resposta = int((vet @ texto.T).squeeze().argmax())''',
        },
        {
            "titulo": "Acima da média",
            "enunciado": "Quantas fotos ficaram com nota acima da média das quatro?",
            "dica": "sum(n > media for n in notas).",
            "solucao": '''media = sum(notas) / len(notas)
resposta = sum(n > media for n in notas)''',
        },
    ],
}

EXERCICIOS = {
    "01-clip-imagem-e-texto.ipynb": AULA1,
    "02-clip-zero-shot-e-busca.ipynb": AULA2,
    "03-difusao-texto-para-imagem.ipynb": AULA3,
    "04-difusao-ruido-e-passos.ipynb": AULA4,
    "05-clip-escolhe-a-imagem.ipynb": AULA5,
}
