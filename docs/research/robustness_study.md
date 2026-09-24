# Estudo exploratório de robustez

## Desenho

A configuração congelada `configs/experiments/robustness_v1.yaml` define a
priori 12 cenários: nominal, feedback aleatório, contraditório e extremamente
negativo, preferências ampla e específica, gênero raro, poucos candidatos,
cauda longa, concentração de popularidade, metadados ausentes e mudança de
catálogo.

O estudo usa exclusivamente os 295 filmes da partição de validação. São 35
agentes independentes (7 personas × 5 seeds oficiais). Em cada cenário, B4 e B5
recebem exatamente o mesmo agente, relevância e conjunto candidato. Nenhuma
célula do holdout oficial é reutilizada e nenhum cenário é selecionado pelo
resultado.

```bash
python -m cinebot_ml.analysis.robustness_execution
```

O estudo válido `56e31e9e624c565ed60a3a64` contém 834 rankings avaliados e 417 pares B4×B5.
Três unidades de `few_candidates` não possuíam qualquer item relevante, logo
NDCG@5 era matematicamente indefinido. Elas foram registradas como falhas e não
foram convertidas em zero. Os demais cenários possuem 35 pares; `few_candidates`
possui 32.

## Inferência

A diferença é B5 menos B4. Cada cenário recebe IC95% bootstrap pareado,
Wilcoxon bilateral, correlação rank-biserial e correção de Holm na família de
12 cenários. A comparação do efeito de cada cenário contra o nominal também é
pareada e corrigida por Holm.

| Cenário | B5 − B4 | IC95% | p-Holm | Classificação |
|---|---:|---:|---:|---|
| contradictory_feedback | -0,2077 | [-0,2891; -0,1280] | 0,000817 | B5 pior |
| random_feedback | -0,0987 | [-0,1816; -0,0277] | 0,2979 | B5 pior pelo IC; inconclusivo após Holm |
| rare_genre | +0,0685 | [0,0133; 0,1331] | 0,3144 | B5 melhor pelo IC; inconclusivo após Holm |
| nominal | +0,0182 | [-0,0404; 0,0682] | 1,000 | inconclusivo |
| long_tail | -0,0401 | [-0,0864; 0,0022] | 1,000 | inconclusivo |
| demais cenários | entre -0,0085 e +0,0926 | intervalos cruzam zero | ≥0,9579 | inconclusivo |

O resultado mais estável é a sensibilidade ao feedback contraditório: B5 perde
cerca de 0,208 NDCG@5 contra B4 e também degrada em relação ao cenário nominal.
Feedback aleatório apresenta sinal adverso: a diferença B5×B4 não permanece
significativa após Holm entre cenários, mas sua degradação frente ao nominal
permanece após Holm (p=0,0157). Ausência de significância não equivale a
robustez demonstrada.

## Limites

A análise é exploratória, sintética e restrita à partição de validação. Personas
não representam grupos humanos. Os resultados diagnosticam a política
incremental congelada; não autorizam modificar B5 ou escolher hiperparâmetros
com base nesses cenários.
