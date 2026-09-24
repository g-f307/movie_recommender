# Métricas de descoberta, popularidade e estabilidade

Este documento fixa as definições implementadas em
`cinebot_ml.ranking.discovery_metrics`. Todas as métricas são calculadas por
unidade experimental antes de qualquer média. As agregações mantêm separados
método, condição, perfil, interação, conjunto candidato, fonte e versão da
relevância.

## Diversidade intra-lista

Cada filme é representado por gêneros, diretores, palavras-chave e década de
lançamento. Para cada par comparável no Top-K, calcula-se a distância de Jaccard:

`d(i,j) = 1 - |F_i ∩ F_j| / |F_i ∪ F_j|`.

A diversidade é a média dessas distâncias e pertence a `[0,1]`. Zero indica uma
lista homogênea e um indica ausência de características em comum. Pares em que
algum item não possui nenhuma característica são ignorados, em vez de receber
distância máxima. A métrica é nula quando não existe par comparável; para um
único item com metadados, vale zero.

## Novidade e exposição à popularidade

A referência de popularidade contém contagens por filme, versão e partição. A
implementação aceita `partition=train` para frequências de interação ou
`partition=catalog_metadata` para metadados externos congelados, como a contagem
de votos usada no estudo da issue #72. Validação e teste comportamental continuam
proibidos como fonte. A identidade da distribuição é registrada em cada resultado.

As probabilidades recebem suavização aditiva: `p_i=(count_i+1)/(Σcount+N)`.
A novidade de um item é sua autoinformação `-log2(p_i)`, normalizada pela maior
autoinformação da distribuição. `novelty@K` é a média e pertence a `[0,1]`.

A exposição converte as contagens em percentis e calcula a média do Top-K, em
`[0,1]`. O viés de popularidade é a exposição do ranking menos a exposição média
do catálogo de referência, em `[-1,1]`: valores positivos indicam concentração
acima da referência e negativos indicam cauda longa.

Se o ranking estiver vazio, a referência estiver ausente ou algum item não
existir na distribuição versionada, essas métricas permanecem nulas.

## Estabilidade entre interações

Rankings sucessivos só são ligados quando possuem a mesma unidade/agente,
método, condição, perfil, seed, K, fonte e versão da relevância, e a interação é
posterior à anterior.

- `ranking_overlap@K`: Jaccard entre os conjuntos recomendados, em `[0,1]`;
- `ranking_repetition@K`: fração de posições com o mesmo item, em `[0,1]`;
- `rank_position_variation@K`: deslocamento absoluto médio dos itens comuns,
  normalizado por `K-1`, em `[0,1]`.

A primeira interação não possui comparação e permanece nula. Rankings vazios
não são convertidos em estabilidade perfeita. Quando não há item comum, a
variação de posição é nula por ausência de observação, enquanto sobreposição e
repetição valem zero.

## Limitações

As métricas descrevem propriedades da lista, não satisfação nem efeito causal.
A novidade depende da população e da janela representadas pela distribuição de
treino. Metadados incompletos reduzem os pares comparáveis; por isso um valor
nulo deve ser preservado, não convertido em zero durante a agregação.
