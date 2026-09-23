# Análise estatística principal B5 × B4

Esta análise implementa o contraste prioritário RQ1/H1 do protocolo v1:
personalização incremental B5 contra o controle estático B4 em NDCG@5.

## Unidade e agregação

A unidade inferencial é o agente sintético independente (agent_id). As
observações de condições e perfis do mesmo agente não são amostras
independentes: primeiro é calculada a média de NDCG@5 por método dentro do
agente, usando P0–P5 e C1–C5; depois os 35 agentes são comparados de forma
pareada. C0 é excluída do teste de benefício porque B5 ainda não recebeu
feedback e, por contrato, equivale a B4.

O CSV preserva uma linha por agente. O JSON contém resumos de B4, B5 e da
diferença B5 − B4, IC de 95% da média por bootstrap pareado com 10.000
reamostragens e seed 42, Wilcoxon bicaudal e correlação rank-biserial.
Empates permanecem contabilizados e o teste de Wilcoxon remove diferenças zero
somente conforme zero_method=wilcox. Valores ausentes ou status inválidos são
contabilizados; pares incompletos não são imputados.

## Integridade

Antes da análise, o carregador confere o status do experimento, identidade da
matriz, dimensões, contagens e hashes do manifesto da matriz e de todas as
células. Arquivos extras, ausentes, alterados ou oriundos de outro diretório
fazem o comando falhar. Isso impede misturar piloto, tentativa interrompida e
execução oficial.

## Reprodução

Com os resultados oficiais recuperados:

    python -m cinebot_ml.analysis

Por padrão são produzidos, sem sobrescrever arquivos existentes:

    results/reports/statistical_analysis.json
    results/tables/b5_vs_b4_ndcg_at_5.csv

Os caminhos podem ser alterados com --manifest, --json-output e --csv-output.

## Limites de interpretação

Os agentes são sintéticos. Portanto, a análise mede o comportamento do sistema
sob o modelo gerador versionado e não demonstra satisfação ou preferência
humana. A análise por subgrupos pertence à issue posterior e será identificada
como exploratória. Ausência de significância não prova a hipótese nula.
