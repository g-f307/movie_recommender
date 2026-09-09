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
.venv/bin/pip install -r requirements-ml.txt
dvc pull
.venv/bin/python -m cinebot_ml.experiment_config validate
.venv/bin/python -m cinebot_ml.experiment_config diagnose
.venv/bin/python -m cinebot_ml.experiment_config prepare
```

`validate` verifica contrato, lock, entradas e escrita das saídas. `diagnose`
mostra somente versões técnicas sanitizadas e estado Git. `prepare` cria a
estrutura `results/{manifests,raw,tables,figures,reports}`. Esses diretórios e
seus resultados são regeneráveis e, portanto, não são versionados.

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
