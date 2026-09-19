# Estudo de ablação A0–A6

A0 preserva o método completo. Cada variante seguinte remove exatamente um
bloco congelado: A1 histórico, A2 feedback negativo, A3 gêneros, A4 diretor,
A5 década e popularidade, e A6 texto. A configuração, o conjunto de componentes
ativos e o identificador de cada variante são versionados.

`AblationContext` mantém agente, seed e `candidate_set_id` constantes. O
executor aplica todas as variantes ao mesmo contexto, exige `ndcg_at_k` e
preserva métricas relacionadas devolvidas pelo runner. A configuração é
rejeitada se uma variante remover bloco diferente do protocolo ou acumular
remoções. Alterações de representação que exijam ajuste devem ser refeitas pelo
runner somente em treino/validação; o holdout não participa dessa escolha.

Os resultados são linhas pareadas por agente e variante, com componentes ativos
e removidos explícitos. O módulo não interpreta significância, cria combinações
extras nem seleciona a melhor variante.
