# Configuração e reprodutibilidade experimental

Este documento define como reconstruir e identificar uma execução do estudo sem
depender da máquina, do diretório corrente ou de parâmetros implícitos.

## Fonte oficial

`configs/experiment_v1.yaml` é a única fonte dos parâmetros científicos da
versão 1: protocolo, métodos, condições, perfis, seeds, valores de K, métricas,
splits e diretórios. O contrato legível por ferramentas está em
`configs/experiment_v1.schema.json`.

A configuração está congelada. Seu SHA-256 consta em
`configs/experiment_v1.lock.json`; qualquer alteração faz a validação falhar.
Uma mudança científica exige novo arquivo versionado (`experiment_v2.yaml`),
novo lock, justificativa no protocolo e revisão por pull request. Nunca se deve
apenas recalcular o lock para ocultar uma alteração.

Credenciais pertencem exclusivamente ao ambiente. Chaves, tokens, conteúdo da
`.env` e caminhos absolutos são proibidos na configuração, em logs e manifests.

## Reconstrução a partir de clone limpo

Na raiz do repositório:

```bash
python -m venv .venv
source .venv/bin/activate
make setup
make validate
make compile
make test
make smoke
```

`make validate` verifica a versão do Python, imports principais, configurações,
schemas, locks e descritores versionados dos dados. Esse fluxo não lê `.env` e
não exige credenciais. `make smoke` executa B0–B5 e C0 em dados mínimos criados
em memória. A CI repete essas verificações em Python 3.11 e 3.12.

Para uma execução completa, use `make readiness`. Além dos contratos, esse modo
verifica catálogo, datasets, remoto DVC e artefatos B2/B3. A ausência ou
incompatibilidade de qualquer ativo gera erro explícito.

### Limitação atual dos ativos

O repositório contém `dvc.yaml` e o ponteiro
`datasets/movie_preferences.csv.dvc`, mas ainda não possui remoto DVC
compartilhado. Portanto, `dvc pull` não recupera os dados em um clone novo.

Existem três caminhos válidos antes da execução oficial:

1. configurar um remoto DVC autorizado e publicar os ativos;
2. regenerar `data/filmes.json` pelo scraper, gerar o dataset e treinar B3 pelos
   estágios do `dvc.yaml`;
3. receber snapshots aprovados e verificar seus hashes antes do uso.

O B2 deve ser ajustado com `fit_tfidf_artifact(..., partition="train")`. O B3
deve possuir metadados do contrato experimental, incluindo partição de ajuste,
partições de seleção e congelamento anterior ao holdout. Artefatos antigos sem
esses campos são rejeitados.

Com o manifesto de split congelado, os comandos oficiais de preparação são:

```bash
make prepare-b2 SPLIT_MANIFEST=results/manifests/splits/<split_id>.json
make prepare-b3
make readiness
```

`prepare-b2` reconstrói as atribuições do manifesto e ajusta o vocabulário e o
IDF somente nos filmes de treino. `prepare-b3` executa o treinamento existente,
que registra partição de ajuste, partições de seleção, schema das features e
congelamento anterior ao holdout.

Depois que os ativos estiverem disponíveis:

```bash
.venv/bin/python -m cinebot_ml.experiment_config validate
.venv/bin/python -m cinebot_ml.experiment_config diagnose
.venv/bin/python -m cinebot_ml.experiment_config prepare
make readiness
```

`prepare` cria `results/{manifests,raw,tables,figures,reports}`. Esses diretórios
e seus resultados são regeneráveis e não são versionados.

O particionamento também carrega esse arquivo por padrão; opções explícitas da
CLI servem apenas para diagnósticos controlados e ficam registradas no manifesto
de split.

Os comandos usam a raiz inferida pelo pacote; funcionam mesmo quando chamados de
outro diretório, desde que o repositório esteja no `PYTHONPATH` ou instalado.

## Manifesto de execução

Antes de cada execução, gere o manifesto:

```bash
.venv/bin/python -m cinebot_ml.experiment_config manifest \
  --method B5 --condition C3 --profile P5 --seed 42 --run 1
```

O arquivo é salvo em `results/manifests/executions/<execution_id>.json` e inclui
cópia lógica da configuração por hash, hashes e versões dos dados, commit Git,
ambiente, parâmetros e caminhos relativos de saída. O `execution_id` deriva
somente desses elementos; duas execuções equivalentes recebem o mesmo ID. O
timestamp é informativo e não participa da identidade.

Exemplo reduzido:

```json
{
  "manifest_version": "1.0",
  "execution_id": "<hash determinístico>",
  "config_sha256": "<hash da configuração normalizada>",
  "method": "B5",
  "condition": "C3",
  "profile": "P5",
  "seed": 42,
  "run": 1,
  "artifacts": {
    "raw": "results/raw",
    "tables": "results/tables",
    "figures": "results/figures",
    "reports": "results/reports"
  }
}
```

Manifests existentes não são sobrescritos com conteúdo diferente. Execuções
finais devem usar worktree limpo; durante desenvolvimento, `worktree_dirty`
permite distinguir evidências não liberáveis.

## Estrutura dos resultados

```text
results/
├── manifests/executions/
├── raw/
├── tables/
├── figures/
└── reports/
```

Somente uma evidência explicitamente congelada e aprovada para publicação deve
ser movida para uma área versionada. Saídas locais permanecem ignoradas.
