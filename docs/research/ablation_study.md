# Estudo de ablação A0–A6

## Desenho

A execução confirmatória usa exclusivamente os 295 filmes da partição de
validação do manifesto `5359277b5b5b14d3`. Nenhum filme, célula ou julgamento
do holdout oficial é reutilizado. Foram definidos 35 agentes independentes
(7 personas × 5 seeds próprias) e cinco condições temporais, totalizando 175
unidades. A0–A6 recebem, dentro de cada unidade, o mesmo agente, contexto,
relevância e `candidate_set_id`.

A0 preserva o sistema incremental completo. A1 remove o histórico; A2, feedback
negativo; A3, gênero; A4, diretor; A5, década e popularidade; e A6, atributos
textuais. O comando reproduzível é:

```bash
make ablation PYTHON=.venv/bin/python
```

Os artefatos são imutáveis e identificados por hashes. O estudo válido é
`538fae9a59ad7976c95c6ef8`, com 1.225 células avaliadas, 175 pares por
comparação e nenhuma célula ausente ou inválida.

## Resultados

A diferença é calculada como variante ablada menos A0 em NDCG@5. Intervalos são
IC95% bootstrap pareados; valores-p vêm do Wilcoxon bilateral e são corrigidos
por Holm na família das seis ablações.

| Variante | Componente removido | Diferença média | IC95% | p-Holm | Efeito rank-biserial | Interpretação |
|---|---|---:|---:|---:|---:|---|
| A1 | histórico | +0,0380 | [0,0188; 0,0579] | 0,00112 | +0,387 | o histórico completo prejudica |
| A2 | feedback negativo | +0,0055 | [-0,0135; 0,0249] | 1,000 | +0,078 | inconclusivo |
| A3 | gênero | -0,1656 | [-0,1913; -0,1403] | <0,000001 | -0,846 | gênero ajuda |
| A4 | diretor | -0,0028 | [-0,0113; 0,0053] | 1,000 | -0,036 | inconclusivo |
| A5 | década e popularidade | -0,0489 | [-0,0718; -0,0259] | 0,000186 | -0,402 | década/popularidade ajudam |
| A6 | texto | +0,0060 | [-0,0066; 0,0187] | 1,000 | +0,095 | inconclusivo |

A0 obteve NDCG@5 médio de 0,6068. A remoção do histórico elevou a média para
0,6449, evidenciando que a política incremental completa, tal como congelada,
introduz perda nesta simulação. Em contraste, gênero e década/popularidade
contribuem positivamente. Feedback negativo, diretor e texto não têm efeito
distinguível de zero neste desenho.

Esses resultados atribuem componentes dentro do ambiente sintético e da
partição de validação; não constituem evidência de comportamento humano nem
autorizam otimização retrospectiva no holdout. Resultados nulos e negativos
foram preservados.
