# Protocolo experimental do Movie Recommender

**Versão:** 1.0
**Status:** candidato a congelamento após revisão do orientador
**Issue:** #2 — `docs(experiments): definir protocolo experimental reproduzível`
**Documento relacionado:** [`research_questions.md`](research_questions.md)

## 1. Finalidade e escopo

Este protocolo estabelece como os métodos de recomendação do Movie Recommender
serão executados, comparados e analisados. Seu objetivo é permitir que outra
pessoa reproduza o estudo sem redefinir condições após observar os resultados.

O protocolo cobre:

- baselines B0–B6;
- condições incrementais C0–C5;
- perfis de cold start P0–P5;
- métricas de ranking, descoberta, estabilidade e eficiência;
- replay temporal;
- ablação e robustez;
- particionamento, repetição, análise estatística e resultados brutos.

Este documento não implementa os métodos nem executa o benchmark. A auditoria
dos dados, a implementação dos splits, a configuração executável e as métricas
serão entregues nas issues técnicas correspondentes.

## 2. Perguntas e prioridade

O estudo responde a RQ1–RQ5 e avalia H1–H5 conforme
[`research_questions.md`](research_questions.md). A ordem de prioridade é:

1. RQ1/H1: efeito do feedback incremental;
2. RQ2/H2: quantidade de feedback necessária;
3. RQ3/H5: relevância versus diversidade e descoberta;
4. RQ4/H3–H4: cold start, baselines e custo;
5. RQ5: contribuição dos componentes.

Se a amostra não sustentar todas as análises, somente RQ1/H1 será confirmatória;
as demais serão identificadas como exploratórias.

## 3. Versão e congelamento

O identificador deste protocolo é `protocol-v1.0`. Após sua homologação pelo
orientador, a configuração correspondente receberá o identificador
`experiment-v1.0`.

Uma mudança incrementará:

- `PATCH`: correção textual que não altera execução ou interpretação;
- `MINOR`: nova análise secundária sem alterar o resultado confirmatório;
- `MAJOR`: alteração de hipótese, métrica primária, split, fonte de relevância,
  método, condição ou critério estatístico principal.

Após o congelamento, nenhuma execução poderá reutilizar o mesmo identificador
com configuração diferente. Exceções deverão registrar justificativa, autor,
data, diff da configuração e novo identificador.

## 4. Dados e fontes de relevância

### 4.1 Catálogo

O catálogo experimental será uma versão imutável de `data/filmes.json`,
identificada por hash SHA-256 e versão DVC. Cada filme deverá possuir `movie_id`
único. Aparições do mesmo filme em mais de um perfil não criarão itens distintos.

Os atributos elegíveis incluem:

- título e sinopse;
- gêneros principal e secundários;
- diretor e palavras-chave;
- ano e período de lançamento;
- nota e quantidade de votos;
- duração, poster e serviços de streaming.

A auditoria posterior definirá quais atributos possuem qualidade suficiente. O
catálogo usado em uma comparação será idêntico para todos os métodos.

### 4.2 Eventos

Cada evento deverá registrar, quando aplicável:

```text
event_id
user_id
session_id
recommendation_timestamp
feedback_timestamp
movie_id
model_name
model_version
protocol_version
experiment_id
seed
rank
score
ranked_genres
decade_preference
popularity_preference
feedback
relevance_label
relevance_source
```

Eventos sem `user_id` ou unidade equivalente não integrarão análises pareadas por
usuário. Eventos sem ordem temporal confiável não integrarão replay incremental.

### 4.3 Hierarquia de evidência

As fontes serão analisadas separadamente:

1. feedback real autorizado;
2. julgamento humano coletado por protocolo;
3. dado público compatível e autorizado;
4. usuário sintético;
5. proxy heurístico.

Conclusões confirmatórias sobre benefício ao usuário exigem uma das três
primeiras fontes. Usuários sintéticos serão usados para validação de engenharia,
robustez e análise exploratória. Proxy heurístico poderá apoiar treinamento e
diagnóstico, mas não comprovará satisfação ou preferência real.

## 5. Perfil, candidatos e relevância

### 5.1 Perfil inicial

O perfil completo contém três gêneros distintos em ordem de preferência,
preferência de década e preferência de popularidade. Valores serão normalizados
pelas mesmas funções para todos os métodos.

### 5.2 Conjunto candidato

Para cada evento `(u,t)`, todos os métodos receberão exatamente o mesmo conjunto
de candidatos. Antes do ranking serão removidos:

- itens indisponíveis ou inválidos na versão congelada do catálogo;
- itens já apresentados a `u` antes de `t`, quando a condição usar histórico;
- itens excluídos por regra de elegibilidade registrada na configuração.

Década e popularidade serão tratadas conforme o método e a condição. Quando uma
comparação exigir filtros estritos, o mesmo filtro será aplicado a todos os
métodos antes do ranking. Se houver menos de `K` candidatos, o evento será
marcado `insufficient_candidates` e tratado conforme a Seção 16.

### 5.3 Relevância

Um item será relevante quando possuir julgamento positivo na fonte de relevância
daquela execução. Escalas não binárias deverão ser convertidas por limiar
definido na configuração antes da execução e preservadas para análises graduadas.

Itens sem julgamento não serão automaticamente considerados negativos. A
estratégia de amostragem e o conjunto de itens julgados deverão acompanhar cada
resultado para controlar viés de exposição.

## 6. Contrato comum dos recomendadores

Todos os métodos deverão implementar semanticamente a operação:

```text
recommend(context, candidates, k) -> ranked_items
```

### Entrada mínima

```json
{
  "experiment_id": "experiment-v1.0__b0__c0__p4__s42",
  "protocol_version": "protocol-v1.0",
  "method": "B0",
  "method_version": "1.0",
  "seed": 42,
  "user_id": "anonymous-stable-id",
  "session_id": "session-stable-id",
  "timestamp": "2026-09-09T00:00:00Z",
  "profile": {
    "ranked_genres": ["acao", "drama", "comedia"],
    "decade_preference": "moderno",
    "popularity_preference": "popular"
  },
  "history": [],
  "candidate_movie_ids": [10, 20, 30],
  "k": 5
}
```

Os identificadores do exemplo são sintéticos e não representam evento real.

### Saída mínima

```json
{
  "experiment_id": "experiment-v1.0__b0__c0__p4__s42",
  "method": "B0",
  "method_version": "1.0",
  "status": "completed",
  "latency_ms": 4.2,
  "ranked_items": [
    {"movie_id": 20, "rank": 1, "score": 0.91},
    {"movie_id": 10, "rank": 2, "score": 0.83}
  ]
}
```

### Invariantes

- `rank` começa em 1 e não se repete;
- `movie_id` não se repete e pertence aos candidatos;
- o ranking possui no máximo `K` itens;
- scores maiores precedem scores menores;
- empates são resolvidos por `movie_id` ascendente, salvo regra versionada;
- método, versão, seed e experimento aparecem na saída;
- falhas usam status explícito e não geram ranking silenciosamente vazio;
- nenhum método recebe informação que outro método da comparação não poderia
  receber, exceto o componente intencionalmente variado.

## 7. Baselines e métodos

| ID | Método | Entrada utilizada | Score e ordenação |
|---|---|---|---|
| B0 | Popularidade | candidatos e filtros permitidos | combinação versionada de votos e nota; desempate por `movie_id` |
| B1 | Content-based | perfil e metadados de item | similaridade entre vetor do perfil e gêneros, diretor, década, popularidade e palavras-chave |
| B2 | TF-IDF + cosseno | texto do perfil, sinopse e gêneros | similaridade de cosseno entre representação textual do perfil e do item |
| B3 | Supervisionado atual | perfil e features textuais, categóricas e numéricas | probabilidade calibrada de aderência; famílias Logistic Regression, Linear SVC calibrado e ComplementNB |
| B4 | Perfil estático | perfil inicial e método base vencedor | ignora feedback posterior e exclui apenas itens conforme regra comum |
| B5 | Incremental | perfil inicial e feedback disponível antes de `t` | combina score base com atualização por feedback positivo e negativo |
| B6 | Híbrido final | componentes autorizados pelo protocolo | combinação definida antes do benchmark final; só será mantido se sua formulação não usar o holdout |

B0–B5 são obrigatórios. B6 é condicional: sua fórmula deverá ser selecionada em
treino/validação e congelada antes do teste final. Se isso não ocorrer, B6 será
omitido e a ausência será registrada.

Para B0, a fórmula exata será configurada e versionada na implementação. Para
B1–B6, hiperparâmetros e pesos serão selecionados somente em treino/validação.

## 8. Condições de feedback

| ID | Histórico disponível antes do ranking |
|---|---|
| C0 | zero feedback |
| C1 | primeira interação concluída |
| C2 | três interações concluídas |
| C3 | cinco interações concluídas |
| C4 | dez interações concluídas |
| C5 | todo o histórico anterior disponível |

Cada condição representa a quantidade máxima de eventos anteriores que o método
pode consultar. O item que receberá feedback no instante atual não integra o
histórico daquele ranking.

Se um usuário não possuir eventos suficientes para C2–C4, a condição será
marcada como indisponível para esse usuário e não será preenchida por repetição
artificial. C5 poderá coincidir com outra condição e essa coincidência será
registrada.

## 9. Perfis de cold start

| ID | Informação disponível |
|---|---|
| P0 | nenhuma preferência declarada |
| P1 | primeiro gênero |
| P2 | três gêneros ordenados |
| P3 | três gêneros e década |
| P4 | três gêneros, década e popularidade |
| P5 | P4 e feedback anterior permitido pela condição C correspondente |

Em P0, o método deverá usar somente informações não pessoais autorizadas, como
popularidade. Campos removidos por uma condição não poderão ser reconstruídos a
partir de outro atributo ou do histórico.

## 10. Valores de K e métricas

### 10.1 Valores oficiais

- `K=5`: valor principal, compatível com a lista entregue pelo bot;
- `K=10`: análise secundária de sensibilidade, quando houver candidatos e
  julgamentos suficientes.

Resultados com outros valores de `K` serão exploratórios e identificados como
tais. As hipóteses serão decididas usando `K=5`.

### 10.2 Métricas primárias

- RQ1, RQ2 e RQ4: NDCG@5;
- RQ3/H5: NDCG@5 e diversidade intra-lista@5;
- RQ5: variação de NDCG@5 em relação ao sistema completo.

### 10.3 Métricas secundárias

- Precision@5 e @10;
- Recall@5 e @10;
- F1@5 e @10;
- NDCG@10;
- MRR;
- diversidade intra-lista;
- novidade;
- cobertura do catálogo;
- viés de popularidade;
- repetição e estabilidade do ranking;
- latência de inferência, tempo de treino, pico de memória e tamanho do artefato.

As fórmulas, intervalos válidos e tratamento de ausência de relevância serão
implementados em módulo único. Um resultado agregado sempre preservará também o
resultado por unidade de análise.

## 11. Unidade de análise

A unidade confirmatória será o usuário independente. Quando não houver usuários
reais e a avaliação usar perfis construídos, a unidade será o perfil e os
resultados serão identificados como offline ou sintéticos.

Para cada unidade e condição, múltiplas sessões serão agregadas primeiro dentro
da unidade. Somente depois serão calculados os resumos entre unidades. Filmes ou
eventos individuais do mesmo usuário não serão tratados como amostras
independentes.

Uma análise confirmatória exigirá pelo menos 30 unidades independentes com dados
válidos na comparação pareada. A meta será 50 unidades. Abaixo de 30, os
resultados serão reportados como exploratórios, com intervalos de confiança e
sem conclusão confirmatória.

## 12. Particionamento e prevenção de leakage

### 12.1 Partições

Os dados supervisionados serão particionados em treino, validação e teste final.
A proporção-alvo será 60%/20%/20% por grupos de `movie_id`, preservando classes
quando tecnicamente possível. A issue de splits poderá ajustar a proporção se a
auditoria demonstrar inviabilidade, exigindo nova versão deste protocolo antes
de observar o teste final.

Feedback temporal será dividido por usuário e tempo. Para cada usuário, somente
eventos anteriores ao ponto de corte poderão compor treino ou histórico. O teste
conterá eventos posteriores e permanecerá isolado.

### 12.2 Proibições

- selecionar modelo, peso, threshold ou regra no teste final;
- usar feedback ocorrido em ou após `t` para produzir `R_m(u,t,K)`;
- deixar o mesmo `movie_id` atravessar partições quando o split exigir isolamento;
- ajustar B6 após observar o holdout;
- reconstruir preferência removida em uma condição de cold start;
- calcular IDF, normalização ou estatística global no conjunto de teste antes da
  transformação pelo pipeline treinado.

Todo split deverá possuir manifesto com versão dos dados, seed, estratégia,
contagens, hashes e verificações de interseção.

## 13. Replay temporal

Para cada usuário, os eventos serão ordenados por `recommendation_timestamp`,
`feedback_timestamp` e `event_id` como desempate estável.

```text
1. carregar perfil inicial e candidatos no instante t;
2. construir h_u(t) somente com eventos anteriores;
3. gerar Top-K com cada método comparável;
4. persistir ranking, scores, versões e latência;
5. obter o julgamento associado ao evento, sem retroagir no ranking;
6. acrescentar o feedback ao histórico;
7. avançar para t+1 e repetir;
8. extrair checkpoints C0–C5 sem reordenar eventos.
```

### Simulação manual C0

1. escolher um perfil P4 e congelar três ou mais candidatos sintéticos;
2. fornecer `history=[]` para B0 e B4;
3. verificar o contrato e gerar Top-5 ou todos os candidatos disponíveis;
4. confirmar que nenhum atributo de feedback aparece na entrada;
5. calcular as métricas apenas se houver julgamentos disponíveis.

### Simulação manual C0 → C1 → C2

1. executar C0 e registrar feedback para o primeiro item apresentado;
2. executar C1 usando apenas esse evento anterior;
3. registrar mais dois feedbacks em ordem temporal;
4. executar C2 usando exatamente três eventos anteriores;
5. executar B4 em paralelo e confirmar que seus scores não mudam por feedback;
6. executar B5 e confirmar que somente eventos anteriores alteram o perfil;
7. comparar os métodos com o mesmo conjunto candidato em cada instante.

## 14. Protocolo de ablação

| ID | Configuração |
|---|---|
| A0 | sistema completo |
| A1 | sem histórico de feedback |
| A2 | somente feedback positivo; feedback negativo indisponível ao método |
| A3 | sem gêneros do perfil e dos itens |
| A4 | sem diretor |
| A5 | sem década e popularidade |
| A6 | sem sinopse e demais features textuais |

Cada ablação modificará somente o componente descrito. Treino, validação e
seleção de hiperparâmetros serão repetidos quando a remoção alterar as features.
Todos os demais dados, candidatos, seeds e condições permanecerão iguais.

A0 será comparado de forma pareada com cada ablação. A análise primária usará
NDCG@5; métricas relacionadas ao componente removido serão reportadas como
secundárias.

## 15. Protocolo de robustez

Serão avaliados, com identificação exploratória:

- feedback aleatório com seed registrada;
- feedback contraditório sobre atributos semelhantes;
- uma única avaliação negativa extrema;
- preferências amplas e específicas;
- poucos candidatos compatíveis;
- gêneros sub-representados;
- concentração em itens populares;
- itens de baixa popularidade;
- campos textuais ou metadados ausentes;
- mudança controlada da distribuição do catálogo.

Para cenários aleatórios serão usadas cinco seeds oficiais:
`42`, `137`, `2027`, `31415` e `65537`. Resultados serão apresentados por seed e
agregados. Métodos determinísticos executarão uma vez por configuração, mas
deverão reproduzir o mesmo resultado nas cinco inicializações.

## 16. Dados ausentes e falhas

| Situação | Tratamento |
|---|---|
| atributo opcional ausente | usar representação ausente definida no pipeline; registrar taxa |
| campo obrigatório ausente | rejeitar evento e registrar `invalid_input` |
| menos de K candidatos | avaliar com lista disponível e marcar `insufficient_candidates`; excluir da análise confirmatória de K quando inviável |
| item sem julgamento | não assumir irrelevância; registrar como não julgado |
| usuário sem histórico suficiente | não imputar condição C; marcar indisponível |
| timestamp inválido | excluir do replay temporal e registrar motivo |
| método falhou | registrar `failed`, erro sanitizado e duração; não substituir por outro método |
| timeout | registrar `timeout` conforme limite configurado |
| execução parcial | preservar resultados válidos e excluir pares incompletos da comparação pareada |

Toda exclusão deverá aparecer em um fluxo de participantes/unidades com contagem
e motivo. Análises de sensibilidade avaliarão o impacto de eventos incompletos
quando houver volume suficiente.

## 17. Repetições, seeds e determinismo

- seeds oficiais: `42`, `137`, `2027`, `31415`, `65537`;
- cinco repetições para métodos ou cenários estocásticos;
- uma execução principal para métodos determinísticos, seguida de verificação de
  determinismo com as cinco seeds;
- bibliotecas relevantes receberão a seed explicitamente;
- ordem de candidatos será normalizada antes de cada execução;
- empate será resolvido conforme o contrato comum;
- ambiente, dependências, hardware e threads serão registrados.

A mesma configuração, dados e seed deverá produzir o mesmo ranking lógico. Se
uma biblioteca impedir determinismo exato, a limitação e a tolerância deverão ser
documentadas antes do benchmark final.

## 18. Análise estatística

### 18.1 Resumo

Para cada método e condição serão reportados número de unidades, média, mediana,
desvio padrão, intervalo interquartil e intervalo de confiança de 95% por
bootstrap pareado com 10.000 reamostragens e seed oficial `42`.

### 18.2 Comparações

- duas condições pareadas: teste de Wilcoxon bicaudal;
- três ou mais condições relacionadas: teste de Friedman;
- pós-testes: Wilcoxon pareado somente após comparação global aplicável;
- correção de múltiplas comparações: Holm dentro de cada família de hipótese;
- nível de significância: `alpha=0,05` após correção;
- tamanho de efeito pareado: correlação rank-biserial, acompanhada da diferença
  bruta e relativa;
- resultados com menos de 30 unidades: exploratórios, independentemente do
  p-valor.

Não serão removidos outliers somente por reduzirem desempenho. Exclusões por erro
de dado deverão obedecer à política da Seção 16 e ser reportadas.

### 18.3 Critérios das hipóteses

- H1/H3/H4: diferença positiva em NDCG@5, `p` corrigido menor que 0,05, IC de 95%
  compatível com ganho e tamanho de efeito reportado;
- H2: Friedman significativo e tendência consistente nos checkpoints; comparações
  pós-teste identificam o primeiro ganho mensurável;
- H5: ganho de NDCG@5 conforme H1 e limite inferior do IC da diferença de
  diversidade superior a `-0,05` ponto absoluto na escala normalizada `[0,1]`.

Se uma condição não atender aos critérios, a hipótese não será sustentada. Isso
não equivale a provar a hipótese nula.

## 19. Convenção de experimentos

O identificador seguirá:

```text
<protocol>__<method>__<condition>__<profile>__s<seed>__r<run>
```

Exemplo:

```text
v1.0__b5__c3__p5__s42__r01
```

Cada execução terá um manifesto imutável contendo:

- identificador e timestamp UTC;
- commit Git e estado do worktree;
- hash da configuração;
- versões do catálogo, dataset e feedback;
- método, versão, condição, perfil, seed e repetição;
- ambiente e dependências;
- status, duração, contagens e caminhos relativos dos artefatos.

Execuções com worktree sujo serão permitidas somente para desenvolvimento e
marcadas `non_release`. Elas não poderão compor os resultados finais.

## 20. Estrutura dos resultados brutos

```text
results/
├── manifests/
│   └── <experiment_id>.json
├── raw/
│   ├── rankings/<experiment_id>.jsonl
│   ├── metrics/<experiment_id>.csv
│   └── failures/<experiment_id>.jsonl
├── tables/
├── figures/
└── reports/
```

O ranking bruto usará um registro por item recomendado. O arquivo de métricas
usará um registro por unidade, método, condição, perfil, seed, repetição, métrica
e valor. Campos mínimos:

```text
experiment_id, protocol_version, method, method_version, condition, profile,
seed, run, unit_type, unit_id, timestamp, k, metric, value, relevance_source,
catalog_version, dataset_version, status
```

Resultados agregados nunca substituirão os resultados por unidade. IDs pessoais
serão pseudonimizados antes de qualquer artefato compartilhável. Segredos,
caminhos absolutos e conteúdo da `.env` são proibidos.

## 21. Verificações antes de cada benchmark

- [ ] protocolo e configuração possuem versão compatível;
- [ ] catálogo, dataset e feedback possuem hash registrado;
- [ ] splits e seus manifestos foram validados;
- [ ] conjunto candidato é idêntico entre métodos comparados;
- [ ] contrato de ranking passa nos testes;
- [ ] nenhuma informação futura aparece no histórico;
- [ ] transformações foram ajustadas somente em treino;
- [ ] seeds e ambiente foram registrados;
- [ ] fonte de relevância está identificada;
- [ ] diretórios de saída estão vazios ou usam novo identificador;
- [ ] worktree está limpo para execução final;
- [ ] dados e artefatos possuem autorização compatível com seu uso.

## 22. Limitações e decisões pendentes

- A fonte principal de relevância da avaliação final ainda depende de dados e
  autorização; sem uma fonte real, humana ou pública compatível, os resultados
  serão exploratórios.
- O tamanho mínimo de 30 unidades é um portão operacional, não substitui análise
  de poder baseada no efeito esperado quando dados piloto estiverem disponíveis.
- O split 60/20/20 poderá ser revisto antes do congelamento se a auditoria mostrar
  classes ou grupos insuficientes.
- A fórmula exata de popularidade de B0 e os pesos de B1/B5/B6 serão definidos em
  treino/validação e registrados na configuração.
- O procedimento de julgamento humano, se adotado, exigirá protocolo de coleta,
  consentimento e governança próprios.
- A publicação de dados, código e resultados depende de autorização
  institucional e de propriedade intelectual.

## 23. Critério de conclusão do protocolo

O protocolo estará pronto para congelamento quando:

- B0–B6, C0–C5 e P0–P5 tiverem implementação ou mapeamento para issues;
- a auditoria confirmar os campos e fontes disponíveis;
- os splits passarem nas verificações contra leakage;
- as métricas tiverem testes com exemplos conhecidos;
- as simulações C0 e C0 → C1 → C2 passarem pelo contrato comum;
- a configuração `experiment-v1.0` representar todas as decisões deste documento;
- o orientador revisar ou registrar solicitações de mudança.
