# Baselines experimentais

Este documento registra as implementações dos métodos de referência B0--B3. A
presente versão contém o baseline B0; os demais serão acrescentados pelas issues
correspondentes sem modificar retroativamente contratos congelados.

## B0 — Popularidade

B0 é um método não personalizado. Ele recebe o `CandidateSet` comum e ordena os
filmes somente por nota agregada e quantidade de votos. Perfil declarado,
histórico e feedback não participam do score.

### Configuração

A especificação oficial está em
`configs/methods/b0_popularity_v1.yaml`, validada por JSON Schema e protegida por
lock SHA-256. Uma mudança de fórmula ou parâmetro exige novo arquivo e nova
versão; o lock atual não deve ser recalculado para alterar resultados já
observados.

### Fórmula

Notas são normalizadas para `[0,1]`. Para cada filme `i`:

```text
score(i) = (v_i / (v_i + m)) × R_i + (m / (v_i + m)) × C
```

Onde:

- `R_i` é a nota normalizada do filme;
- `v_i` é sua quantidade de votos;
- `C` é a média das notas válidas do conjunto candidato;
- `m` é o quantil 0,60 da quantidade de votos dos candidatos.

Essa média bayesiana reduz o efeito de avaliações extremas sustentadas por
poucos votos. Conforme `v_i` cresce, o score se aproxima da nota do item; com
poucos votos, aproxima-se da média do catálogo elegível.

O cálculo de `C` e `m` usa somente o conjunto candidato daquela execução. Ambos
são registrados no manifesto do método.

### Valores ausentes

- nota ausente, inválida ou fora de `[0,10]`: substituída por `C`;
- votos ausentes, inválidos ou negativos: substituídos por zero;
- nenhum candidato: ranking concluído e vazio;
- menos de K candidatos: todos são devolvidos, sem repetição artificial.

Esses fallbacks mantêm o conjunto comum intacto. B0 não remove unilateralmente
um filme que outro baseline receberia.

### Ordenação

O score é ordenado de forma decrescente. Empates usam `movie_id` ascendente,
conforme o contrato comum. O resultado inclui score bruto, componentes do
cálculo, versão do método e `candidate_set_id`.

### Uso

```python
from cinebot_ml.ranking import PopularityRecommender

request = candidate_set.attach_to_request(initial_request)
recommender = PopularityRecommender(candidate_set)
result = recommender.recommend(request)
method_manifest = recommender.method_manifest()
```

`method_manifest` registra a configuração congelada, seu hash, `C`, `m` e a
identidade do conjunto candidato. Ele pode ser fornecido a
`build_execution_manifest` para compor a identidade reproduzível da execução.

### Limitações e viés

B0 tende a favorecer filmes que já acumularam avaliações, reforçando exposição
e popularidade histórica. Votos e notas do TMDB refletem a população e o período
da coleta, não necessariamente os usuários do experimento. A média bayesiana
controla incerteza de poucos votos, mas não elimina esse viés.

Por não usar preferências, B0 não mede personalização. Sua função é estabelecer
uma referência simples e forte: métodos mais complexos só demonstram valor se
superarem essa estratégia sob os mesmos candidatos e a mesma fonte de
relevância.
