# Baselines experimentais

Este documento registra as implementações dos métodos de referência B0--B3. A
presente versão contém os baselines B0 e B1; os demais serão acrescentados pelas
issues correspondentes sem modificar retroativamente contratos congelados.

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
