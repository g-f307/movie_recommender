# Decisão metodológica v1 sobre B6

**Issue:** #47. **Decisão registrada na preparação do experimento v1:** adiar
B6; manter B0–B5. Este documento explicita a justificativa daquela decisão,
sem propor um método após observar os resultados oficiais.

## Comparação conceitual e hipótese

| Aspecto | B5 implementado | B6 descrito no roadmap |
| --- | --- | --- |
| Entrada | Perfil inicial, candidatos e feedback anterior ao ranking | Nenhuma entrada adicional especificada |
| Mecanismo | Score estático de B4 acrescido de ajuste por afinidades aprendidas | Apenas “modelo híbrido final” |
| Hipótese distinguível | Efeito do feedback incremental contra B4 | Nenhuma hipótese própria definida |
| Componente novo | Atualização incremental já implementada | Nenhum componente identificável |

O termo *híbrido* não distingue B6 de B5: B5 já combina perfil estático e
feedback. Sem entrada, mecanismo e comparação distintos, não há contraste
falsificável específico para B6. A hipótese “B6 supera B5” seria apenas uma
expectativa de desempenho, não uma especificação do método a testar. Por isso,
**não definimos uma hipótese, fórmula nem pesos artificiais para B6 na v1**.

## Custo, risco e decisão

Implementar um B6 distinto exigiria especificar componente novo, desenvolver e
testar o recomendador, selecionar parâmetros somente em treino/validação,
congelar uma nova configuração e acrescentar células pareadas à matriz. O custo
experimental e de revisão aumentaria; não há estimativa defensável de ganho.
Renomear B5 custaria menos, mas criaria uma contribuição fictícia. Definir B6
depois de ver o holdout introduziria seleção pós-resultado. Logo, a decisão
para a versão 1 é **adiar**, não implementar ou incluir B6 no artigo como
método avaliado.

## Rastreabilidade da versão executada

`configs/experiment_v1_holdout.yaml` mantém `B6` em `methods.optional` como
possibilidade do schema, mas `methods.enabled` contém somente B0–B5. A matriz
oficial não possui células B6 e seu manifesto registra
`b6: deferred_no_distinct_method`. Esses arquivos e seus hashes integram a
versão congelada; **não se altera a configuração do holdout já executado** para
retirar o nome de `optional`, pois isso quebraria a identidade da evidência.
O protocolo registra expressamente a decisão v1.1, preservando a regra geral
para versões futuras.

Uma versão futura só poderá propor B6 com hipótese, arquitetura, custo,
configuração, plano de seleção em treino/validação e novo teste independente
definidos antes de consultar seus resultados. Os resultados finais da v1 não
servirão como teste confirmatório de uma formulação posterior. Não há issues
técnicas de B6 a criar enquanto a decisão for de adiamento.

Esta é uma decisão de escopo da issue #47, **não** uma aprovação do orientador.
A ausência de revisão externa e de evidência humana deve permanecer nas
limitações do artigo.
