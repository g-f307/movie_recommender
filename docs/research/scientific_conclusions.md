# Conclusões científicas, decisões e ameaças à validade

**Versão:** 1.0
**Issue:** #62 — `docs(scientific-conclusions): decidir H1–H5 e registrar limitações`
**Matriz oficial:** `2942e51456add29e4c999307`
**Evidência:** exploratória, produzida por agentes sintéticos

## Escopo

Este documento decide H1–H5 e responde RQ1–RQ5 com a execução oficial v1.1. A unidade inferencial é o agente sintético independente, não cada célula. Os resultados descrevem o sistema sob o gerador versionado; não demonstram satisfação, preferência ou benefício humano.

A única análise confirmatória é B5 contra B4 em NDCG@5. Cold start, convergência e subgrupos são exploratórios. Ablação e robustez permanecem indisponíveis. Uma média favorável isolada não sustenta hipótese.

## Matriz RQ → hipótese → evidência → decisão

| RQ | Hipótese | Evidência reproduzível | Decisão |
|---|---|---|---|
| RQ1 | H1 — benefício do feedback | B5 − B4=−0,01809; IC95% [−0,03427; −0,00222]; Wilcoxon p=0,03562; rank-biserial=−0,40635; 9 ganhos, 26 perdas; n=35 | **Não sustentada:** o sinal é contrário. |
| RQ2 | H2 — aprendizado progressivo | Friedman C0–C5: χ²=9,3473; p=0,09599; n=35. B5 contra C0: C1 −0,01314; C2 −0,01793; C3 −0,01611; C4 −0,01575; C5 +0,01070 | **Não sustentada:** sem tendência positiva global ou convergência demonstrada. |
| RQ4 | H3 — mitigação do cold start | B5: P0=0,32515 e P5=0,62711; B4: P0=0,29258 e P5=0,67379. Sem contraste inferencial P5 × P0 | **Inconclusiva:** sinal descritivo favorável sem evidência inferencial exigida. |
| RQ4 | H4 — personalização versus popularidade | B5=0,53987 e B0=0,34708; n=35. Sem teste pareado B5 × B0 ou análise conjunta de descoberta/custo | **Inconclusiva:** evidência mínima não produzida. |
| RQ3 | H5 — relevância sem perda de diversidade | O ganho de relevância falhou em H1; margem de não inferioridade não congelada; novidade e viés de popularidade indisponíveis | **Não sustentada:** falhou condição necessária e não inferioridade não foi testada. |
| RQ5 | contribuição por ablação | Nenhuma execução válida A0–A6; o holdout não foi reinterpretado como ablação | **Inconclusiva:** componentes individuais não podem ser responsabilizados. |

Em H1 não se aplica correção por multiplicidade: é a comparação primária única. Nos subgrupos, Holm foi aplicado por dimensão e K. Em H2, o teste global não rejeitou a hipótese nula e não houve pós-testes. Em H3/H4, IC, p corrigido e efeito são `N/D`: os contrastes exigidos não existem e não são inferidos de médias.

## Respostas às perguntas

### RQ1 — efeito do feedback incremental

B5 reduziu NDCG@5 frente a B4 em 0,01809, ou 3,25% da média de B4. O IC não cruza zero, o teste atinge 5% e o efeito é moderado e negativo. Há significância estatística, mas a magnitude prática média é pequena. Como 26 de 35 agentes perderam, o sinal não deve ser tratado como mero ruído.

### RQ2 — quantidade de feedback

Não surgiu quantidade de interações com melhora estável. B5 ficou abaixo de C0 entre C1 e C4 e teve ganho descritivo apenas em C5. A curva não é monotônica e o teste global não foi significativo. O salto C4→C5 é indício exploratório.

### RQ3 — relevância versus descoberta

O trade-off não foi decidido. A relevância caiu, não há margem congelada para diversidade, e novidade/viés de popularidade estão ausentes. Diversidade favorável, se observada, não resgataria H5 sem ganho de relevância.

### RQ4 — qualidade, complexidade e custo

O ranking descritivo foi B1/B4 (0,55494), B5 (0,53987), B3 (0,52297), B2 (0,45977) e B0 (0,34708). Sugere valor do perfil e vantagem de B5 sobre B0 no simulador, mas não prova H3/H4: faltam contrastes e métricas comparáveis de latência, memória, artefato e treino. Não há “melhor relação qualidade–custo”.

### RQ5 — contribuição dos componentes

Não há resposta empírica. A0–A6 foram especificadas, mas não executadas em estudo separado. Essa ausência evita selecionar explicações pelo holdout.

## Heterogeneidade e explicações plausíveis

Os subgrupos são diagnósticos exploratórios. Após Holm, permaneceram sinais desfavoráveis em C3 e P3–P5 em K=5, além de P3 em K=10. Em P5/K=5, B5−B4 foi −0,05603, IC95% [−0,08714; −0,02671], p bruto=0,00183, p de Holm=0,00930 e rank-biserial=−0,64113. Só a seed 42 favoreceu B5; as outras quatro foram negativas. Personas têm n=5 e seeds n=7, sem precisão para conclusões isoladas.

Uma explicação compatível é reação excessiva a feedback escasso ou ruidoso, deslocando um perfil inicial informativo. O prejuízo em P3–P5 reforça essa possibilidade. É mecanismo a testar, não causa demonstrada. Também são plausíveis desalinhamento com a função latente, exposição limitada e sensibilidade à seed.

## Ameaças à validade

### Interna

- B4/B5 foram pareados e as 15.120 células validadas por hash.
- Método, feedback e relevância compartilham o domínio sintético; essa circularidade pode favorecer ou punir regras sem representar causalidade.
- Sem ablação e robustez, componentes e mecanismos não podem ser isolados.

### Construto

- NDCG@5 sintético mede alinhamento com preferências latentes, não satisfação.
- Agentes são arquétipos, não pessoas ou grupos demográficos; seus eventos não são feedback real.
- Novidade e viés estão ausentes; diversidade não tem margem congelada.
- O catálogo filtrado pode reduzir cauda longa, ambiguidade e casos reais.

### Externa

- Não houve validação humana e o feedback real é escasso. Nada deve ser generalizado para usuários do Telegram, outras populações ou catálogos.
- Personas e seeds não cobrem toda mudança de gosto e contexto.

### Conclusão

- H1 respeita pareamento, IC, teste e efeito; p=0,03562 não transforma uma diferença pequena em grande efeito prático.
- H2–H5 têm lacunas e foram classificadas conservadoramente.
- Subgrupos continuam exploratórios após Holm; persona e seed têm n pequeno.
- Ausências não foram imputadas e resultados negativos não foram removidos.

## Contribuição e enquadramento do artigo

A contribuição sustentável não é “feedback sempre melhora recomendação”. É a infraestrutura reproduzível e a evidência negativa de que atualização ingênua pode degradar um perfil estático forte, sobretudo quando ele já é informativo. Isso permanece relevante sem superioridade de B5 e enquadra o artigo em avaliação experimental, simulação controlada e falhas de personalização.

O artigo deve separar engenharia, achado confirmatório no simulador, achados exploratórios e hipóteses não testadas. Não deve afirmar “usuários preferem”, “satisfação aumentou” ou “o método é melhor” sem evidência humana.

## Trabalho futuro prioritário

1. executar A0–A6 separadamente do holdout;
2. executar robustez, sobretudo feedback contraditório, aleatório e extremo;
3. congelar margem e medir diversidade, novidade, cobertura e popularidade;
4. produzir contrastes P5 × P0 e B5 × B0 e custos de RQ4;
5. investigar regularização, taxa, esquecimento e confiança em treino/validação;
6. validar em dados públicos e, quando viável, protocolo humano autorizado.

## Rastreabilidade e reprodução

Fontes: [`research_questions.md`](research_questions.md), [`experimental_protocol.md`](experimental_protocol.md), [`statistical_analysis.md`](statistical_analysis.md), [`subgroup_analysis.md`](subgroup_analysis.md), [`experiment_synthesis.md`](experiment_synthesis.md), [`scientific_reporting.md`](scientific_reporting.md) e [`artifact_reproduction.md`](artifact_reproduction.md).

```bash
python -m cinebot_ml.analysis.reporting --output-root results/scientific_report_v1
```

Conferir `scientific_report.json` e as tabelas B5×B4, testes, subgrupos, cold start, convergência e métodos no diretório gerado.
