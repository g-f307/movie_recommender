# Particionamento experimental e prevenção de leakage

## Objetivo

O módulo `cinebot_ml.experimental_splits` centraliza os splits utilizados pela
pesquisa. Ele separa grupos, controla a ordem temporal do feedback, protege o
holdout final e persiste manifestos determinísticos.

O módulo não executa treinamento nem escolhe modelos. Sua saída define quais
grupos ou eventos cada etapa do pipeline poderá consumir.

## Estratégias implementadas

### Isolamento por grupo

O split agrupado aceita qualquer coluna de grupo. Os usos previstos são:

- `movie_id`: impede que o mesmo filme atravesse treino, validação e teste;
- `user_id`: isola usuários quando o protocolo exigir generalização para novos
  usuários;
- identificador de perfil ou contexto: disponível para estudos adicionais,
  desde que definido na configuração experimental.

O particionamento usa proporção 60/20/20 e busca deterministicamente, entre 128
candidatos derivados da seed, a distribuição com menor divergência de tamanho e
classe. As partições permanecem não vazias e todos os grupos aparecem exatamente
uma vez.

Datasets com uma única classe são rejeitados, pois não permitem verificar a
preservação das classes prevista pelo protocolo.

Identificadores de usuário devem ser pseudonimizados antes de entrar no módulo.
O manifesto preserva os IDs de grupo para permitir reconstrução e, por isso, não
deve receber identificadores pessoais ou IDs brutos de plataformas externas.

### Isolamento temporal

Eventos são agrupados por usuário e ordenados por timestamp UTC. Timestamps
iguais do mesmo usuário permanecem na mesma partição. Para cada usuário:

```text
eventos antigos → treino → validação → teste → eventos recentes
```

Um usuário precisa possuir ao menos três timestamps distintos para compor as
três partições. Timestamps ausentes, inválidos ou sem fuso horário são rejeitados
antes da geração do manifesto.

A função `history_before` retorna somente eventos estritamente anteriores ao
instante da recomendação. O evento atual e feedbacks futuros não entram no
histórico.

### Proteção do holdout

`assert_partition_allowed` rejeita o uso da partição `test` para:

- seleção de modelo;
- seleção de threshold;
- calibração;
- ajuste de hiperparâmetros.

O teste final é permitido apenas para avaliação final depois que método,
threshold, pesos e hiperparâmetros estiverem congelados.

## Comandos

Gerar os manifestos padrão a partir do dataset e feedback atuais:

```bash
python -m cinebot_ml.experimental_splits
```

Escolher seed e diretório de saída:

```bash
python -m cinebot_ml.experimental_splits \
  --seed 42 \
  --attempts 128 \
  --output-dir results/manifests/splits
```

O comando recusa sobrescrever arquivos existentes. Uma substituição consciente
exige `--overwrite` e deve ser acompanhada de nova versão ou justificativa.

Os manifestos locais em `results/manifests/splits/` são regeneráveis e ficam
ignorados pelo Git durante o desenvolvimento. A versão definitiva deverá ser
congelada e rastreada junto de `experiment-v1.0`, conforme a política que será
definida na configuração experimental.

## Contrato do manifesto agrupado

Exemplo reduzido e sintético:

```json
{
  "manifest_id": "identificador-derivado-do-conteudo",
  "split_version": "1.0",
  "strategy": "deterministic_group_stratification_search",
  "source_name": "dataset.csv",
  "source_sha256": "sha256-da-fonte",
  "group_column": "movie_id",
  "label_column": "relevante",
  "config": {
    "seed": 42,
    "train_fraction": 0.6,
    "validation_fraction": 0.2,
    "test_fraction": 0.2,
    "attempts": 128,
    "protocol_version": "protocol-v1.0"
  },
  "total_groups": 5,
  "total_rows": 10,
  "partitions": {
    "train": {
      "group_count": 3,
      "rows": 6,
      "class_counts": {"0": 3, "1": 3},
      "group_ids": ["m1", "m2", "m3"],
      "group_ids_sha256": "sha256-dos-ids"
    }
  }
}
```

O manifesto real contém as três partições completas. Os IDs são necessários para
reconstruir a atribuição; o hash detecta alteração do conteúdo.

## Contrato do manifesto temporal

Exemplo reduzido e sintético:

```json
{
  "manifest_id": "identificador-derivado-do-conteudo",
  "split_version": "1.0",
  "strategy": "per_user_temporal",
  "source_name": "feedback.csv",
  "source_sha256": "sha256-da-fonte",
  "total_events": 5,
  "partitions": {
    "train": {
      "event_count": 3,
      "event_indexes": [0, 1, 2],
      "event_indexes_sha256": "sha256-dos-indices",
      "minimum_timestamp": "2026-01-01T00:00:00+00:00",
      "maximum_timestamp": "2026-01-03T00:00:00+00:00"
    }
  }
}
```

As três partições armazenam índices e limites temporais. A reconstrução valida os
hashes e exige cobertura exata dos eventos.

## Reconstrução e validação

Para manifestos agrupados:

```python
from pathlib import Path
from cinebot_ml.experimental_splits import (
    load_manifest,
    reconstruct_assignments,
    validate_manifest_source,
)

manifest = load_manifest(Path("results/manifests/splits/movie_id_split.json"))
validate_manifest_source(manifest, Path("datasets/movie_preferences.csv"))
assignments = reconstruct_assignments(manifest)
```

Se o dataset mudar, `validate_manifest_source` interrompe a execução. Se um ID
aparecer em mais de uma partição, se faltar um grupo ou se um hash não conferir,
a reconstrução também falha.

## Evidência da execução atual

Uma execução local com seed 42 e 128 tentativas produziu:

| Partição | Filmes | Linhas | Classe 0 | Classe 1 |
|---|---:|---:|---:|---:|
| Treino | 886 | 1.116.360 | 983.250 | 133.110 |
| Validação | 295 | 371.700 | 327.413 | 44.287 |
| Teste | 295 | 371.700 | 327.433 | 44.267 |

O split temporal dos 14 eventos disponíveis resultou em 8 eventos de treino, 3
de validação e 3 de teste. O relatório expõe somente contagens e intervalos de
tempo, sem identificadores pessoais.

## Garantias verificadas

- mesma configuração e seed produzem o mesmo manifesto;
- seeds diferentes podem produzir atribuições diferentes;
- nenhum grupo aparece em duas partições;
- timestamps iguais não atravessam partições;
- feedback futuro não integra histórico passado;
- alterações na fonte invalidam o manifesto;
- seleção ou calibração no holdout são bloqueadas;
- manifestos existentes não são sobrescritos silenciosamente;
- splits agrupados e temporais podem ser reconstruídos.

## Limitações

- A busca preserva a proporção de classes por aproximação; não garante igualdade
  exata entre partições.
- O split por usuário somente será executável quando o dataset experimental
  possuir quantidade suficiente de usuários e classes.
- O feedback atual possui apenas um usuário e não sustenta avaliação
  confirmatória.
- A política definitiva de rastreamento dos manifestos será congelada com a
  configuração `experiment-v1.0`.
