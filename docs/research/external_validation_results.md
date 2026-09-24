# Validação externa no MovieLens 100K

**Versão:** 1.0
**Issue:** #73 — external validation
**Protocolo:** [external_validation_protocol.md](external_validation_protocol.md)
**Configuração:** `configs/experiments/external_validation_v1.json`

## Fonte e separação

O estudo utiliza o MovieLens 100K, publicado pelo GroupLens, com 100.000 avaliações, 943 usuários e 1.682 filmes. O arquivo oficial foi validado pelo MD5 `0e33842e24a9c977be4e0107933c0723`. Os dados brutos não são redistribuídos: o comando de reprodução os obtém da fonte oficial e registra a auditoria.

A divisão cronológica global, congelada antes da análise, produziu 70.000 avaliações de treino, 10.000 de adaptação e 20.000 de teste. Nenhum evento de adaptação ou teste participa do perfil inicial, e nenhum evento de teste atualiza B5.

## Cobertura e comparabilidade

Todas as avaliações foram mapeadas. Aplicando os critérios pré-especificados — ao menos cinco eventos de treino, um de adaptação e um item relevante no teste — 43 de 943 usuários foram elegíveis (4,56%). Essa baixa cobertura limita a generalização e não foi corrigida após observar os dados.

As versões públicas de B4 e B5 usam somente gêneros porque o MovieLens não contém preferências declaradas, condições, personas e contexto do simulador. B0 é popularidade bayesiana calculada apenas no treino. A validação é parcial; as amostras nunca são combinadas.

## Resultados

| Contraste NDCG@5 | Diferença média | IC95% | p bilateral | Ganhos / perdas / empates |
|---|---:|---:|---:|---:|
| B5 − B4 | 0,00000 | [0,00000; 0,00000] | 1,00000 | 0 / 0 / 43 |
| B5 − B0 | −0,07848 | [−0,12893; −0,03407] | 0,00428 | 1 / 13 / 29 |

B4 e B5 obtiveram média 0,01465; B0 obteve 0,09313. A correlação rank-biserial B5−B0 foi −0,86667. A cobertura do catálogo foi 1,90% para B0, 3,21% para B4 e 2,91% para B5.

O empate integral B5−B4 é neutro. O efeito negativo sintético (−0,01809; IC95% [−0,03427; −0,00222]) não foi replicado, mas também não foi contradito por efeito positivo. A decisão é `inconclusive_neutral_external`.

## Impacto sobre RQ1–RQ5

- **RQ1/H1:** não sustentada no simulador; externamente inconclusiva.
- **RQ2/H2:** não transportável sem C0–C5.
- **RQ3/H5:** não transportável sem os mesmos construtos de descoberta.
- **RQ4/H3:** não transportável sem P0–P5.
- **RQ4/H4:** a superioridade sintética sobre popularidade não se transportou.
- **RQ5:** as ablações não possuem componentes semanticamente idênticos.

## Reprodução

```bash
make final-evidence
```

Os derivados são rastreados por DVC no remoto privado. O ZIP bruto permanece fora do Git e do DVC em respeito às condições de uso.

## Referência

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* 5, 4, Article 19. https://doi.org/10.1145/2827872
