# Análise exploratória de subgrupos

Esta análise complementa o contraste principal B5 × B4 e procura identificar
em quais condições do simulador o feedback ajuda, é neutro ou prejudica o
ranking. Seus resultados são exploratórios e não substituem a análise
confirmatória principal.

## Pareamento e unidade

O pipeline recupera os resultados oficiais somente após validar seus hashes.
Cada linha detalhada preserva os IDs das células B4 e B5, agente, persona,
seed, condição, perfil e K. Antes de calcular qualquer teste, observações
repetidas são agregadas dentro do mesmo agente sintético.

- condição: agrega os seis perfis dentro de cada agente;
- perfil: agrega C1–C5 dentro de cada agente;
- persona: agrega C1–C5 e P0–P5 dentro de cada agente;
- seed: agrega C1–C5 e P0–P5 dentro de cada agente.

C0 participa apenas da análise por condição, pois ainda não contém feedback.
K=5 e K=10 são analisados separadamente; K=10 é sensibilidade. Personas possuem
cinco agentes e seeds sete agentes, abaixo do mínimo de 30, e recebem alerta
explícito de baixa precisão.

## Inferência exploratória

Cada subgrupo registra média de B4 e B5, diferença B5 − B4, IC bootstrap de
95%, ganhos, perdas, empates, Wilcoxon bicaudal e correlação rank-biserial. A
correção de Holm é aplicada separadamente para cada dimensão e K. Mesmo quando
o valor-p ajustado é inferior a 0,05, o resultado continua exploratório.

Todos os grupos são emitidos; o pipeline não seleciona somente resultados
favoráveis. Os dez maiores ganhos e perdas preservam IDs que permitem retornar
às células oficiais.

## Reprodução

    python -m cinebot_ml.analysis.subgroups

O comando gera, sem sobrescrita:

    results/reports/subgroup_analysis.json
    results/tables/subgroup_summary.csv
    results/tables/subgroup_pairs.csv
    results/figures/subgroup_heatmap_k5.png
    results/figures/subgroup_heatmap_k10.png
    results/figures/subgroup_distributions_k5.png
    results/figures/subgroup_distributions_k10.png

## Resultado observado na execução v1.1

Foram reconstruídos 2.520 pares, sem métricas inválidas ou pares incompletos.
Após correção de Holm, os sinais mais claros permaneceram desfavoráveis ao B5:
C3 em K=5; P3, P4 e P5 em K=5; e P3 em K=10. Personas e seeds continuam úteis
para diagnóstico de heterogeneidade, mas suas amostras pequenas não sustentam
conclusões isoladas.
O sinal não foi estável em todas as seeds: a seed 42 favoreceu B5 em K=5 e
K=10, enquanto as outras quatro produziram diferenças médias negativas. Esse
contraste exige investigação e não autoriza selecionar apenas a seed favorável.

O maior ganho e a maior perda individuais mostram que há grande variação entre
contextos. Esses extremos são casos para inspeção, não estimativas do efeito
médio. Personas são comportamentos simulados e não grupos demográficos humanos.
