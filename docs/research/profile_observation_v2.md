# Perfil declarado observável v2

## Propósito

Na execução v1.1, a preferência usada para gerar a utilidade sintética também
era projetada diretamente nos perfis P0--P5. A execução complementar v2 separa
essas funções: a preferência latente permanece no agente e um perfil declarado
imperfeito é a única informação estática entregue a B4 e B5.

Essa separação torna explícito que uma preferência declarada é uma observação
do construto latente, não uma cópia dele. Ela não transforma a simulação em
evidência humana nem altera os artefatos oficiais v1.1.

## Configuração e determinismo

`configs/agents/profile_observation_v2.yaml` contém a política congelada de
observação. O JSON Schema exige seis intensidades no intervalo `[0, 1]` e o
lock SHA-256 impede alterar a configuração sem criar nova versão.

| Campo observado | Perturbação configurada |
| --- | --- |
| gêneros | omissão e alteração da ordem declarada |
| década | classificação em outra faixa temporal |
| popularidade | classificação na categoria oposta |
| diretores | omissão de valores declarados |
| palavras-chave | omissão de valores declarados |

A projeção é determinística para a combinação de `agent_id`, seed e hash da
configuração. Ela não consulta ranking, feedback, posição, score, resultados
do holdout ou eventos futuros.

## Limite de informação

`DeclaredPreference.to_payload()` produz somente `ranked_genres`,
`decade_preference`, `popularity_preference`, `preferred_directors` e
`preferred_keywords`. A regra P0--P5 continua determinando qual subconjunto
pode ser usado: P0 não recebe preferências, P1 recebe um gênero, P2 até três,
P3 adiciona década, P4 adiciona popularidade e P5 adiciona diretores,
palavras-chave e o histórico causal permitido.

O estado inicial entregue aos recomendadores não contém `preference_id`, pesos
de utilidade ou atributos latentes não declarados. A função que gera utilidade,
julgamento e relevância continua usando apenas `LatentPreference`.

## Rastreabilidade

Para execução v2, cada snapshot e manifesto registra uma proveniência externa
ao perfil entregue ao método: `latent_preference_id`, `declared_profile_id`,
fonte, seed da observação, hash da configuração e intensidades de ruído. Isso
permite auditar a relação entre as representações sem disponibilizar a
preferência latente a B4/B5.

`configs/experiments/execution_v2.yaml` referencia essa configuração como
insumo da execução, mas não a trata como diretório de saída. A v1.1 não recebe
essa referência e preserva suas identidades e artefatos originais.

## Verificação

```bash
python -m unittest \
  tests.test_synthetic_agents \
  tests.test_cold_start_study \
  tests.test_experiment_units
```

Os testes verificam schema e lock, determinismo, perfil sem ruído, máscaras
P0--P5, proveniência e ausência de identificadores latentes no payload de
requisição.
