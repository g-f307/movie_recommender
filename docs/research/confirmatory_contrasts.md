# Contrastes confirmatórios H3 e H4

**Issue:** #71 — `feat(confirmatory-contrasts): implementar inferência P5 × P0 e B5 × B0`
**Matriz oficial:** `2942e51456add29e4c999307`
**Unidade inferencial:** agente sintético independente
**Desfecho principal:** NDCG@5; NDCG@10 é sensibilidade

## Protocolo

H3 compara P5 contra P0 sob B5 nas condições pós-feedback C1–C5. H4 compara B5 contra B0 com C0 separado de C1–C5. Repetições são agregadas dentro de cada agente antes da inferência. Todo par exige a mesma identidade experimental e o mesmo `candidate_set_id`; pares ausentes são contabilizados e nunca imputados.

A família confirmatória contém H3/P5−P0/K=5 e H4/B5−B0/pós-feedback/K=5. Os valores-p bilaterais de Wilcoxon são corrigidos por Holm. A decisão positiva requer diferença média na direção prevista, limite inferior do IC95% bootstrap acima de zero, p de Holm menor que 0,05 e pelo menos 30 agentes.

## H3 — perfil completo contra ausência de perfil

Em 35 agentes, P5 superou P0 sob B5 em `0,28188` NDCG@5, IC95% `[0,22880; 0,33677]`, p de Holm `2,33 × 10⁻¹⁰` e correlação rank-biserial `0,99683`. Em K=10, o ganho foi `0,28740`, IC95% `[0,24358; 0,33389]`.

A curva média em K=5 foi P0=`0,33616`, P1=`0,51423`, P2=`0,51989`, P3=`0,62740`, P4=`0,61306` e P5=`0,61805`. Somente 6 de 35 agentes apresentaram curva integralmente não decrescente. Logo, o contraste extremo P5×P0 sustenta H3 no simulador, mas os perfis intermediários não formam uma progressão monotônica.

**Decisão:** H3 sustentada para relevância sintética, sem generalização para preferência humana.

## H4 — método personalizado contra popularidade

Após feedback, B5 superou B0 em `0,18983` NDCG@5, IC95% `[0,13468; 0,24853]`, p de Holm `8,63 × 10⁻⁸` e correlação rank-biserial `0,91111`. Em C0, o ganho foi `0,20762`, IC95% `[0,15951; 0,26008]`. A sensibilidade em K=10 preservou a direção: `0,21997` após feedback e `0,22366` em C0.

**Decisão:** o componente de relevância de H4 é sustentado, mas a hipótese completa permanece condicionada à avaliação de descoberta, cobertura e viés de popularidade. A vantagem sobre B0 não deve ocultar a deterioração de B5 contra B4 já encontrada em H1.

## Limites de interpretação

- Os 35 agentes são sintéticos; os resultados não medem satisfação humana.
- H3 demonstra diferença entre extremos, não monotonicidade entre P0–P5.
- H4 compara relevância; descoberta e cobertura são necessárias para sua decisão final.
- B5 pode superar popularidade e ainda perder para um perfil estático forte.
- K=10 é sensibilidade e não substitui o desfecho pré-especificado em K=5.

## Reprodução

```bash
make contrasts PYTHON=.venv/bin/python
```

A execução gera `contrasts.json`, `p5_vs_p0.json`, `b5_vs_b0.json`, CSVs por agente e um manifesto com hashes em `results/derived/specialized_v1/contrasts/<matrix_id>/`. As saídas são imutáveis: um diretório já existente não é sobrescrito.
