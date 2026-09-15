# Baselines experimentais

Este documento registra as implementações dos métodos de referência B0--B3. A
presente versão contém os baselines B0, B1, B2 e B3. Métodos posteriores serão
acrescentados pelas issues correspondentes sem modificar retroativamente
contratos congelados.

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

## B1 — Conteúdo estruturado

B1 é o primeiro baseline personalizado. Ele compara preferências declaradas
com os metadados estruturados dos filmes do `CandidateSet`, sem consultar ou
aprender com histórico e feedback. Isso isola o valor da informação inicial do
perfil antes da personalização incremental.

### Configuração e seleção dos pesos

A especificação oficial está em `configs/methods/b1_content_v1.yaml`, validada
por JSON Schema e protegida por lock SHA-256. A versão 1 usa peso unitário para
cada bloco de informação e pesos `[1,00; 0,65; 0,35]` para os três gêneros
ordenados.

Esses valores são uma política *a priori*, não o resultado de otimização. A
implementação não ajusta vocabulário, normalização ou pesos aos dados e não usa
o holdout. Uma seleção futura em treino/validação deverá gerar nova versão da
configuração, sem recalcular o lock da v1 depois de observar o teste.

### Informação autorizada por perfil

| Perfil | Componentes usados por B1 |
| --- | --- |
| P0 | nenhum; todos os itens recebem score zero |
| P1 | primeiro gênero declarado |
| P2 | até três gêneros ordenados |
| P3 | P2 e década |
| P4 | P3 e preferência de popularidade |
| P5 | P4 e, quando declarados estaticamente, diretores e palavras-chave |

Campos de níveis posteriores são ignorados em perfis anteriores. B1 nunca
deriva preferências do `history`; assim, likes, dislikes e eventos passados não
alteram seu ranking. Diretores e palavras-chave de P5 devem vir do perfil
inicial autorizado, não de reconstrução retrospectiva do feedback.

### Representação e fórmula

Gêneros, diretores e palavras-chave são conjuntos canônicos; década e faixa de
popularidade são categorias. Para gêneros, o componente é a fração dos pesos
ordenados cujos gêneros aparecem no filme. Década e popularidade usam igualdade
binária. Diretores e palavras-chave usam Jaccard entre preferências e item.

Para os componentes ativos `A`:

```text
score(i, p) = Σ[a ∈ A] peso(a) × similaridade(a, i, p) / Σ[a ∈ A] peso(a)
```

A normalização considera apenas atributos liberados e declarados pelo perfil.
Isso permite comparar níveis de cold start sem punir P1 por não possuir década,
por exemplo. Em P0, `A` é vazio e o score é zero; o desempate canônico por
`movie_id` garante uma saída determinística.

### Metadados ausentes e prevenção de leakage

Metadado ausente produz similaridade zero apenas no componente correspondente,
sem remover o filme do conjunto comum. Um ano ausente não é convertido
artificialmente em “antes de 2000”. Listas ausentes de palavras-chave e diretor
ausente são representados por conjuntos vazios.

As transformações são determinísticas e não possuem etapa de ajuste. O método
não acessa eventos futuros, não altera candidatos e não usa feedback. A
identidade da configuração, o snapshot, a política sem fit, a proibição de uso
do holdout e o `candidate_set_id` são registrados pelo `method_manifest`.

### Uso

```python
from cinebot_ml.ranking import ContentRecommender

request = candidate_set.attach_to_request(initial_request)
recommender = ContentRecommender(candidate_set)
result = recommender.recommend(request)
method_manifest = recommender.method_manifest()
```

### Limitações

B1 depende da cobertura e qualidade dos metadados. Similaridade estruturada não
captura nuances semânticas da sinopse, e preferências declaradas podem não
refletir o comportamento real. O método também pode concentrar resultados em
categorias majoritárias. Sua função experimental é servir como referência
estática e interpretável para medir o ganho posterior de métodos textuais,
híbridos e incrementais.

## B2 — TF-IDF e similaridade de cosseno

B2 mede a contribuição de uma representação textual esparsa. O texto de cada
filme combina sinopse, gêneros canônicos e palavras-chave; o texto do perfil é
formado somente pelos gêneros declarados autorizados pelo nível de cold start.
Histórico e feedback incremental não participam do método.

### Configuração congelada

A especificação oficial está em `configs/methods/b2_tfidf_v1.yaml`, validada
por JSON Schema e protegida por lock SHA-256. A versão 1 usa:

- conversão para minúsculas e remoção Unicode de acentos;
- unigramas e bigramas;
- até 5.000 atributos;
- `min_df=1`, frequência sublinear e norma L2;
- nenhuma lista de stop words;
- fallback de vetor zero para documentos vazios;
- descarte de termos ausentes do vocabulário.

Esses parâmetros são congelados antes da avaliação. Alterá-los requer uma nova
versão; o lock da v1 não deve ser recalculado após observar resultados finais.

### Ajuste e prevenção de leakage

O vocabulário e os valores de IDF são ajustados exclusivamente por
`fit_tfidf_artifact(..., partition="train")`. Qualquer tentativa de ajuste com
`validation` ou `test` falha explicitamente. Candidatos de validação ou teste
são apenas transformados pelo artefato previamente construído.

O artefato JSON contém:

- vocabulário e índices canônicos;
- valores de IDF na ordem dos índices;
- versão do método e hash da configuração;
- hash do corpus de treino;
- versão do scikit-learn;
- `artifact_id` derivado de todo o conteúdo.

O carregamento recalcula a identidade e rejeita artefatos adulterados. Como o
arquivo é regenerável, sua localização operacional deve permanecer em uma
pasta de resultados ignorada pelo Git; configuração, schema e código são os
elementos versionados.

### Perfil textual

| Perfil | Texto utilizado por B2 |
| --- | --- |
| P0 | vazio |
| P1 | primeiro gênero |
| P2–P5 | até três gêneros ordenados |

Os gêneros recebem repetições `[3, 2, 1]`, preservando a ordem de preferência
antes do cálculo de TF-IDF. Década, popularidade, texto livre injetado e
histórico não são utilizados por B2 v1; por isso, esses campos não conseguem
reconstruir informação removida por um perfil de cold start.

### Score e ordenação

Para perfil `p` e filme `i`:

```text
score(i, p) = cos(TFIDF(p), TFIDF(i))
```

Como os vetores usam norma L2, o score pertence a `[0,1]` para as frequências
não negativas usadas pelo método. Perfil vazio, item vazio ou perfil composto
somente por termos desconhecidos recebe similaridade zero. Empates seguem o
contrato comum e usam `movie_id` ascendente.

### Uso

```python
from cinebot_ml.ranking import TfidfRecommender, fit_tfidf_artifact

artifact = fit_tfidf_artifact(training_movies, partition="train")
artifact.save(generated_artifact_path)

request = candidate_set.attach_to_request(initial_request)
recommender = TfidfRecommender(candidate_set, artifact)
result = recommender.recommend(request)
method_manifest = recommender.method_manifest()
```

O manifesto registra configuração, artefato, hash do corpus, tamanho do
vocabulário, versão da biblioteca, partição de ajuste e `candidate_set_id`.

### Limitações

TF-IDF captura coincidência lexical, mas não equivalência semântica entre
sinônimos ou conceitos relacionados. A ausência de stop words em português e o
limite de atributos podem manter ruído ou descartar termos raros. Além disso, o
perfil textual v1 é deliberadamente restrito aos gêneros declarados; isso torna
B2 uma referência controlada, não uma representação completa dos interesses do
usuário.

## B3 — Supervisionado

B3 adapta o pipeline supervisionado existente ao contrato comum de ranking. O
score de cada candidato é a probabilidade prevista para a classe positiva de
aderência ao perfil. Todos os candidatos recebem score; o threshold binário não
remove itens e não participa da ordenação.

### Natureza da supervisão

Os rótulos atuais são majoritariamente derivados do proxy heurístico de
aderência, com sobrescritas pontuais quando existe feedback explícito. Portanto,
os scores de B3 não representam probabilidade validada de preferência humana.
O resultado e o manifesto carregam o aviso e a origem
`heuristic_proxy_with_explicit_feedback_overrides`.

Métricas como accuracy, F1 e ROC-AUC permanecem diagnósticos da tarefa de
classificação do proxy. Elas não substituem NDCG, Precision, Recall, MAP ou MRR
no benchmark de recomendação.

### Configuração e contrato do artefato

A especificação oficial está em `configs/methods/b3_supervised_v1.yaml`,
validada por JSON Schema e protegida por lock SHA-256. O artefato aceito por B3
deve possuir:

- modelo com `predict_proba` e classe positiva `1`;
- schema de features exatamente igual ao código versionado;
- família, parâmetros e threshold selecionados;
- partição de fit igual a `train`;
- partições de seleção iguais a `train` e `validation`;
- confirmação de congelamento antes do holdout;
- origem dos rótulos explicitamente declarada.

O carregamento calcula SHA-256 do modelo e dos metadados. Ambos aparecem no
manifesto, junto com configuração, família, parâmetros, threshold, features e
`candidate_set_id`.

### Seleção sem holdout

O fluxo de treino avalia as famílias Logistic Regression, Linear SVC calibrado
e ComplementNB por validação cruzada agrupada dentro da porção de treino. A
família, os hiperparâmetros e o threshold são escolhidos pelas métricas dessa
validação e congelados. Só então o vencedor é ajustado no treino e observado
uma única vez no holdout para diagnóstico final.

O teste final não participa da escolha entre famílias. As funções de guarda
bloqueiam `fit` fora de treino e seleção no holdout.

### Perfil e features ausentes

B3 libera preferências progressivamente:

| Perfil | Preferências fornecidas ao modelo |
| --- | --- |
| P0 | nenhuma; campos do perfil recebem `__missing__` |
| P1 | primeiro gênero |
| P2 | até três gêneros |
| P3 | P2 e década |
| P4–P5 | P3 e popularidade; feedback é ignorado por B3 |

Campos proibidos de níveis posteriores são ignorados e não podem reconstruir a
informação removida. Categorias desconhecidas são mantidas para o
`OneHotEncoder(handle_unknown="ignore")`. Texto e categorias ausentes usam
string vazia; números ausentes, inválidos ou infinitos usam zero. Isso preserva
o candidato em vez de removê-lo unilateralmente.

### Ranking

```text
score(i, p) = P(modelo; classe aderente = 1 | features(i, p))
```

Scores não finitos ou fora de `[0,1]` provocam falha explícita. Valores válidos
são ordenados de forma decrescente, com desempate por `movie_id`. O threshold
selecionado aparece somente em `score_components.diagnostic_threshold`.

### Uso

```python
from cinebot_ml.ranking import SupervisedArtifact, SupervisedRecommender

artifact = SupervisedArtifact.load(model_path, metadata_path)
request = candidate_set.attach_to_request(initial_request)
recommender = SupervisedRecommender(candidate_set, artifact)
result = recommender.recommend(request)
method_manifest = recommender.method_manifest()
```

### Situação do artefato legado

O modelo produzido em junho de 2026 escolheu a família vencedora comparando
métricas no conjunto de teste. Seus metadados também não registram o schema e o
congelamento exigidos pela versão B3. Por esse motivo, o adaptador o rejeita
deliberadamente. Ele precisa ser retreinado pelo fluxo corrigido antes de um
benchmark B3 válido; renomear seus metadados não resolveria o leakage já
ocorrido.

### Limitações

Mesmo após o retreino correto, B3 aprende principalmente uma fórmula proxy que
já codifica correspondências entre gêneros, década e popularidade. O desempenho
de classificação muito alto pode refletir essa circularidade, não qualidade de
recomendação. B3 será uma referência supervisionada no estudo, e suas alegações
devem permanecer limitadas até avaliação Top-K com uma fonte de relevância
adequada.
