# Rastreabilidade do manuscrito ACM SAC

**Issue:** #80 — consolidação do manuscrito
**Matriz oficial:** `2942e51456add29e4c999307`
**Estado:** versão consolidada para revisão; empacotamento final pertence à #81

## Decisões editoriais

- O achado central é negativo e limitado ao simulador: B5 reduziu NDCG@5 contra B4.
- H1, H2 e H5 não são sustentadas; H3 e H4 são sustentadas somente no domínio sintético.
- Convergência, subgrupos, ablação e robustez permanecem exploratórios.
- A margem de não inferioridade de diversidade é `−0,05` e foi congelada antes da execução.
- MovieLens é validação parcial e inconclusiva; as amostras nunca são combinadas.
- NDCG sintético não é descrito como satisfação, preferência observada ou benefício humano.

## Elementos e fontes

| Elemento | Alegação sustentada | Fonte estruturada |
|---|---|---|
| Figura 1 | distribuição B5−B4 e IC95% | análise estatística oficial |
| Tabela 1 | contraste primário e tamanho de efeito | análise estatística oficial |
| Figura 2 | trajetória descritiva C0–C5 | síntese experimental |
| Figura 3 | ablação A1–A6 e robustez | estudos especializados em validação |
| Tabela 2 | H1, H3, H4 e contraste externo | contrastes e validação externa |
| Figura 4 | separação sintético × MovieLens | relatório de transportabilidade |

Os arquivos em `paper_artifacts/` são gerados por `make paper-artifacts` e não são editados manualmente. O manifesto editorial registra hashes das fontes e saídas.

## Verificações antes da submissão

A issue #81 deverá confirmar anonimização, referências, limite oficial de páginas, metadados do PDF, checksums e compilação em clone limpo.
