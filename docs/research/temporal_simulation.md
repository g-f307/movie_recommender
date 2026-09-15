# Simulação temporal C0–C5

O executor temporal coordena recomendação, apresentação, feedback sintético e
atualização do perfil em uma sequência reproduzível. Seu propósito é medir o
estado do ranking após quantidades controladas de feedback sem permitir que um
evento influencie retroativamente a recomendação que o originou.

## Condições e checkpoints

A configuração congelada `configs/simulation_v1.yaml` define:

| Condição | Eventos anteriores disponíveis no ranking final |
| --- | ---: |
| C0 | 0 |
| C1 | 1 |
| C2 | 3 |
| C3 | 5 |
| C4 | 10 |
| C5 | todo o histórico configurado, 15 eventos na v1 |

C5 representa o histórico integral da trajetória configurada e pode ser
alterada somente em uma nova versão da configuração. A política seleciona o
primeiro item apresentado para feedback; não existir candidato encerra a
trajetória como `exhausted`, sem fabricar eventos.

## Ordem temporal

Para cada interação `N`, o executor:

1. constrói candidatos excluindo itens já consumidos;
2. gera o ranking usando apenas o snapshot de entrada;
3. apresenta o Top-K e seleciona o item da política de feedback;
4. solicita o julgamento ao agente sintético;
5. registra um `FeedbackEvent` com timestamp posterior ao ranking;
6. cria um novo snapshot imutável;
7. avança o relógio lógico.

Depois da quantidade de eventos da condição, um ranking final é calculado com
o histórico permitido. Assim, o feedback do passo N aparece somente no estado e
nos rankings posteriores. B4 sempre recebe o perfil inicial; B5 recebe o
snapshot vigente.

## Resultado e retomada

`SimulationScenario` identifica método, condição, perfil, seed, catálogo,
estado inicial e relógio. `SimulationStep` preserva snapshot de entrada,
candidatos, itens apresentados, ranking, julgamento, evento e snapshot de
saída. `SimulationResult` agrega os passos, ranking final, manifesto e eventual
falha.

O identificador da execução deriva do cenário e das versões das políticas, não
da latência observada. Checkpoints `partial`, `exhausted` e `failed` podem ser
persistidos atomicamente. Uma retomada aceita somente resultado do mesmo
cenário e continua a partir do último snapshot válido.

O método `run_many` aceita pares de cenário e agente, permitindo executar
múltiplas seeds e personas. Comparações pareadas devem fornecer a mesma seed,
agente e snapshot inicial aos métodos.

## Execução

No diretório do projeto:

```bash
.venv/bin/python -m cinebot_ml.simulation \
  --method B5 \
  --condition C2 \
  --profile P4 \
  --persona consistent \
  --seed 42 \
  --k 5
```

Por padrão, o checkpoint regenerável é escrito em
`results/raw/simulations/<scenario_id>.json`, diretório fora do versionamento.
Também é possível informar `--catalog` e `--output`. O arquivo registra hashes,
seeds, agente, políticas, eventos, estados e rankings necessários para auditar a
trajetória.

## Limitações

A trajetória depende da função de utilidade sintética e da política que escolhe
o item apresentado. Ela permite testar causalidade temporal e comportamento do
sistema, mas não demonstra satisfação ou generalização para usuários humanos.
Latência é uma medição operacional e pode variar entre reexecuções; estados,
eventos, rankings, identificadores e decisões pseudoaleatórias permanecem
reproduzíveis para entradas equivalentes.
