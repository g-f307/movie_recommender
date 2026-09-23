# Log sanitizado de reprodução v1

- Data: 2026-09-23
- Commit testado: `2c24fbc`
- DVC: 3.67.1
- Armazenamento: remoto local descartável, sem credenciais
- Clone: novo clone Git criado com `--no-local`

## Resultado

- 17.648 objetos/arquivos enviados ao remoto de validação.
- 17.648 objetos/arquivos recuperados no clone.
- Cinco ativos do inventário aprovados por tamanho, hash e ponteiro.
- Manifesto oficial `2942e51456add29e4c999307` aprovado.
- 15.120 células oficiais revalidadas.
- Seis tabelas e quatro figuras regeneradas no clone.
- Tabelas e figuras do clone foram idênticas byte a byte às do ambiente de origem.

## Hashes dos artefatos científicos regenerados

```text
245b033f02e5c2271f45ee0606b35ef519d2a6019c1a8e92eb5ac9176514edcc  figures/ablation.png
555fc6a8224f09ca7d0acdc59aae707f98feaf82f00f65b47b69c81658cbada9  figures/cold_start.png
5dc43071bf2d764e6b83268fd70a072e9d1a9719436b8d819a80ecce522edd85  figures/convergence.png
89114fed0f2f22afa74328ae24bb531daf1744a76c0c6c960fc995161b18ed12  figures/robustness.png
ace18952ba6aca39fe067bf05b76e8d3609c23b5bb176dc87eb9906368206c83  tables/b5_vs_b4_primary.csv
fec320e99e435ac5496da034430f91c2c2d54d4577e500834fd1b96118c0f22d  tables/cold_start.csv
46a24aa5cae26c22433f4f00414ce89c94bc4e1e3631e6a6b159ed2487cb76f7  tables/convergence.csv
6fda903d4d3fd00565090a33d2053eef8c588c2449c06cc91a18fc810a387256  tables/methods_b0_b5.csv
b716162778c323c76fcc0e7ca5bd5326e3dbd16c5bb878df988e1bf8b8cf4989  tables/statistical_tests.csv
a03b71955ee1a63666212b5fed8bd12c260d7b4e8aed1f5049f8ee2b265d48b1  tables/subgroups.csv
```

## Limitações observadas

- Não havia backend DVC compartilhado autorizado; a validação usou armazenamento temporário.
- O ponteiro legado do dataset de preferências está desatualizado e foi excluído deste fluxo.
- Feedback real, catálogo e credenciais não foram copiados para o remoto.
- A validação comprova reprodutibilidade computacional dos resultados sintéticos, não replicação com participantes humanos.
