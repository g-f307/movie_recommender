# Estudo de cold start P0–P5

## Objetivo

O estudo mede como a disponibilidade progressiva de informação afeta B0–B5,
sem alterar a preferência latente do agente. Não realiza o teste estatístico de
H3 e seus resultados sintéticos não representam usuários humanos.

## Projeções de perfil

As projeções são congeladas em `configs/experiments/cold_start_v1.yaml`:

| Perfil | Informação autorizada |
|---|---|
| P0 | nenhuma preferência declarada |
| P1 | primeiro gênero |
| P2 | até três gêneros |
| P3 | P2 e década |
| P4 | P3 e preferência de popularidade |
| P5 | P4, diretores, palavras-chave e histórico causal permitido por C0–C5 |

Campos fora da lista de cada perfil são rejeitados. P0–P4 recebem estado de
versão zero e não podem reconstruir histórico, década, popularidade, diretores
ou palavras-chave por atributos indiretos. P5 recebe uma cópia do estado causal
produzido antes do ranking avaliado; eventos futuros continuam proibidos pelo
contrato temporal.

## Pareamento

Uma trajetória de referência é produzida uma vez para o mesmo agente, persona e
seed. As seis projeções preservam `agent_id` e `preference_id`. Dentro de cada
perfil, todos os métodos recebem o mesmo `CandidateSet`, julgamentos sintéticos
e estado permitido. Resultados continuam identificados por método, perfil,
condição, agente, seed, K e versão do estado.

## Saídas

Cada execução possui `study_id` determinístico e produz, sem sobrescrita:

```text
<output>/<study_id>/
├── cold_start.manifest.json
├── cold_start.individual.jsonl
├── cold_start.availability.csv
└── cold_start.summary.csv
```

O JSONL é o dataset bruto por agente, método e perfil. O resumo contém somente
médias descritivas preliminares. A tabela de disponibilidade registra
`insufficient_candidates` quando o conjunto possui menos de K itens; esse caso
não é descartado nem transformado em falha silenciosa.

## Reprodução

Exemplo com métodos que não exigem artefatos externos:

```bash
python -m cinebot_ml.experiments cold-start \
  --catalog data/filmes.json \
  --condition C2 --persona consistent --seed 42 --k 5 \
  --method B0 --method B1 --method B4 --method B5 \
  --output results/raw/cold_start
```

Para B2, informe `--b2-artifact`. Para B3, informe `--b3-model` e
`--b3-metadata`. Seed, K, métodos, perfis e suas configurações são validados
antes da execução.

## Limitações

O resumo não corrige dependência entre interações e não deve ser usado como
teste confirmatório. A trajetória e os julgamentos são sintéticos; o estudo
quantifica comportamento sob o simulador versionado, não eficácia clínica ou
preferência humana real.
