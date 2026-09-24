# Conclusões científicas, decisões e ameaças à validade

**Versão:** 1.1
**Issue:** #62 — `docs(scientific-conclusions): decidir H1–H5 e registrar limitações`
**Matriz oficial:** `2942e51456add29e4c999307`
**Evidência:** confirmatória e exploratória sintética, com validação externa parcial no MovieLens 100K

## Escopo

Este documento decide H1–H5 e responde RQ1–RQ5 com a execução oficial v1.1. A unidade inferencial é o agente sintético independente, não cada célula. Os resultados descrevem o sistema sob o gerador versionado; não demonstram satisfação, preferência ou benefício humano.

As análises confirmatórias incluem B5 contra B4 em NDCG@5 e os contrastes pré-especificados P5×P0 e B5×B0. Convergência, perfis intermediários, ablação e robustez são exploratórios. Ablação e robustez foram executadas somente em validação e não alteram retrospectivamente os testes confirmatórios. Uma média favorável isolada não sustenta hipótese.

## Matriz RQ → hipótese → evidência → decisão

| RQ | Hipótese | Evidência reproduzível | Decisão |
|---|---|---|---|
| RQ1 | H1 — benefício do feedback | B5 − B4=−0,01809; IC95% [−0,03427; −0,00222]; Wilcoxon p=0,03562; rank-biserial=−0,40635; 9 ganhos, 26 perdas; n=35 | **Não sustentada:** o sinal é contrário. |
| RQ2 | H2 — aprendizado progressivo | Friedman C0–C5: χ²=9,3473; p=0,09599; n=35. B5 contra C0: C1 −0,01314; C2 −0,01793; C3 −0,01611; C4 −0,01575; C5 +0,01070 | **Não sustentada:** sem tendência positiva global ou convergência demonstrada. |
| RQ4 | H3 — mitigação do cold start | P5−P0 sob B5=+0,28188; IC95% [0,22880; 0,33677]; Holm p=2,33×10⁻¹⁰; rank-biserial=0,99683; n=35 | **Sustentada no simulador:** P5 supera P0, sem monotonicidade garantida entre perfis intermediários. |
| RQ4 | H4 — personalização versus popularidade | B5−B0 pós-feedback=+0,18983 em NDCG@5; novidade 0,61091×0,41594; cobertura 0,97627×0,02373; latência mediana 80,19×18,70 ms | **Sustentada no simulador:** B5 supera B0 em qualidade e descoberta, com maior custo. |
| RQ3 | H5 — relevância sem perda de diversidade | Diversidade B5−B4=−0,00553; IC95% [−0,00937; −0,00172]; margem=−0,05; p unilateral=2,91×10⁻¹¹. Relevância=−0,01809 | **Não sustentada:** diversidade é não inferior, mas falhou o ganho simultâneo de relevância. |
| RQ5 | contribuição por ablação | Em validação, remover gênero reduziu NDCG@5 em 0,1656 e remover década/popularidade reduziu 0,0489; remover histórico elevou 0,0380. Todos com IC95% sem zero e Holm aplicado | **Parcialmente sustentada:** gênero e década/popularidade ajudam, enquanto o histórico incremental completo prejudica neste simulador. |

Em H1 não se aplica correção por multiplicidade por ser a comparação primária original. H3 e o componente de relevância de H4 formam uma família corrigida por Holm. Nos subgrupos, Holm foi aplicado por dimensão e K. Em H2, o teste global não rejeitou a hipótese nula e não houve pós-testes.

## Respostas às perguntas

### RQ1 — efeito do feedback incremental

B5 reduziu NDCG@5 frente a B4 em 0,01809, ou 3,25% da média de B4. O IC não cruza zero, o teste atinge 5% e o efeito é moderado e negativo. Há significância estatística, mas a magnitude prática média é pequena. Como 26 de 35 agentes perderam, o sinal não deve ser tratado como mero ruído.

### RQ2 — quantidade de feedback

Não surgiu quantidade de interações com melhora estável. B5 ficou abaixo de C0 entre C1 e C4 e teve ganho descritivo apenas em C5. A curva não é monotônica e o teste global não foi significativo. O salto C4→C5 é indício exploratório.

### RQ3 — relevância versus descoberta

B5 preservou diversidade frente a B4 e ampliou novidade e cobertura, mas reduziu NDCG@5. A margem de não inferioridade de −0,05 foi congelada antes da execução e atendida; ainda assim, H5 não é sustentada porque o ganho simultâneo de relevância era condição necessária. O resultado caracteriza um trade-off, sem score composto para mascará-lo.

### RQ4 — qualidade, complexidade e custo

P5 superou P0 sob B5 em 0,28188 NDCG@5, sustentando mitigação de cold start no simulador, embora somente 6 de 35 agentes tenham exibido progressão integralmente não decrescente entre P0 e P5. B5 também superou B0 após feedback em 0,18983, com resultado preservado em K=10.

H4 é sustentada no simulador: contra B0, B5 melhora relevância, novidade, cobertura e viés de popularidade, mas sua latência mediana é 80,19 ms contra 18,70 ms. Frente a B4, B5 ganha cobertura e novidade, mantém diversidade dentro da margem, custa mais e perde relevância.

Não há “melhor relação qualidade–custo” universal. Os seis métodos permaneceram na fronteira multidimensional de Pareto porque preservam compromissos distintos; escolher um campeão exigiria pesos externos previamente justificados.

### RQ5 — contribuição dos componentes

O estudo separado A0–A6 atribuiu parte do resultado: gênero e década/popularidade contribuem positivamente, enquanto o histórico incremental completo degrada o ranking. Feedback negativo, diretor e texto permaneceram inconclusivos. A execução usou somente validação, preservando o holdout.

## Heterogeneidade e explicações plausíveis

Os subgrupos são diagnósticos exploratórios. Após Holm, permaneceram sinais desfavoráveis em C3 e P3–P5 em K=5, além de P3 em K=10. Em P5/K=5, B5−B4 foi −0,05603, IC95% [−0,08714; −0,02671], p bruto=0,00183, p de Holm=0,00930 e rank-biserial=−0,64113. Só a seed 42 favoreceu B5; as outras quatro foram negativas. Personas têm n=5 e seeds n=7, sem precisão para conclusões isoladas.

A ablação e a robustez tornam mais específica a explicação: remover todo o histórico melhora o ranking e feedback contraditório causa perda média de 0,2077 em NDCG@5 frente a B4. Feedback aleatório também apresenta sinal adverso, embora não sobreviva à correção de Holm entre cenários. Isso sustenta sensibilidade da atualização a sinais inconsistentes dentro do simulador, não uma causa generalizável para pessoas.

## Ameaças à validade

### Interna

- B4/B5 foram pareados e as 15.120 células validadas por hash.
- Método, feedback e relevância compartilham o domínio sintético; essa circularidade pode favorecer ou punir regras sem representar causalidade.
- Ablação e robustez isolam mecanismos em validação sintética, mas não eliminam confundimento do gerador nem validam causalidade humana.

### Construto

- NDCG@5 sintético mede alinhamento com preferências latentes, não satisfação.
- Agentes são arquétipos, não pessoas ou grupos demográficos; seus eventos não são feedback real.
- Novidade e viés dependem dos votos do catálogo congelado como proxy externa; a margem de diversidade foi uma escolha substantiva prévia de cinco pontos percentuais.
- O catálogo filtrado pode reduzir cauda longa, ambiguidade e casos reais.

### Externa

- Não houve validação humana e o feedback real é escasso. Nada deve ser generalizado para usuários do Telegram, outras populações ou catálogos.
- Personas e seeds não cobrem toda mudança de gosto e contexto.

### Conclusão

- H1 respeita pareamento, IC, teste e efeito; p=0,03562 não transforma uma diferença pequena em grande efeito prático.
- H2 e H5 não foram sustentadas; H3 e H4 foram sustentadas somente no domínio sintético avaliado.
- Subgrupos continuam exploratórios após Holm; persona e seed têm n pequeno.
- Ausências não foram imputadas e resultados negativos não foram removidos.

## Contribuição e enquadramento do artigo

A contribuição sustentável não é “feedback sempre melhora recomendação”. É a infraestrutura reproduzível e a evidência negativa de que atualização ingênua pode degradar um perfil estático forte, sobretudo quando ele já é informativo. Isso permanece relevante sem superioridade de B5 e enquadra o artigo em avaliação experimental, simulação controlada e falhas de personalização.

O artigo deve separar engenharia, achado confirmatório no simulador, achados exploratórios e hipóteses não testadas. Não deve afirmar “usuários preferem”, “satisfação aumentou” ou “o método é melhor” sem evidência humana.

## Validação externa e transportabilidade

A avaliação pré-especificada no MovieLens 100K usou separação cronológica 70/10/20, sem compartilhar dados nem ajustar hiperparâmetros a partir do holdout sintético. Dos 943 usuários, 43 atenderam aos critérios de elegibilidade. B5−B4 foi exatamente 0,00000 em NDCG@5 (IC95% [0,00000; 0,00000]; 43 empates), enquanto B5−B0 foi −0,07848 (IC95% [−0,12893; −0,03407]; p=0,00428).

O efeito externo neutro não replica nem contradiz com efeito favorável o resultado sintético. As versões externas baseadas em gênero não reproduzem P0–P5, C0–C5 ou personas. A decisão é transportabilidade parcial e inconclusiva: H1 segue não sustentada no simulador; H2, H3 e H5 não são semanticamente transportáveis; e a superioridade sintética de H4 sobre popularidade não se repetiu no recorte público.

## Trabalho futuro prioritário

1. investigar regularização, taxa, esquecimento e confiança para explicar a perda B5×B4;
2. validar o mecanismo em outro conjunto público com histórico mais denso ou protocolo semanticamente mais próximo da elicitação inicial;
3. repetir custos em hardware reportável se comparações de desempenho forem centrais;
4. quando viável, executar protocolo humano autorizado.

## Rastreabilidade e reprodução

Fontes: [`research_questions.md`](research_questions.md), [`experimental_protocol.md`](experimental_protocol.md), [`statistical_analysis.md`](statistical_analysis.md), [`confirmatory_contrasts.md`](confirmatory_contrasts.md), [`discovery_and_cost.md`](discovery_and_cost.md), [`subgroup_analysis.md`](subgroup_analysis.md), [`experiment_synthesis.md`](experiment_synthesis.md), [`scientific_reporting.md`](scientific_reporting.md) e [`artifact_reproduction.md`](artifact_reproduction.md).

```bash
python -m cinebot_ml.analysis.reporting --output-root results/scientific_report_v1
```

Conferir `scientific_report.json` e as tabelas B5×B4, testes, subgrupos, cold start, convergência e métodos no diretório gerado.
