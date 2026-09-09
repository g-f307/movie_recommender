# Perguntas de pesquisa, hipóteses e variáveis

**Documento:** definição científica do Movie Recommender
**Versão:** 1.0
**Status:** candidato à revisão do orientador
**Issue:** #1 — `docs(science): formalizar perguntas de pesquisa, hipóteses e variáveis`

## 1. Finalidade

Este documento define o que será investigado na evolução científica do Movie
Recommender. Ele delimita as perguntas de pesquisa, as hipóteses testáveis, as
variáveis e os resultados mínimos necessários para interpretar os experimentos.

O objeto de estudo não é a existência de um chatbot de filmes. O objeto de
estudo é o efeito da personalização incremental sobre rankings de recomendação
em diferentes condições de cold start.

As definições deste documento serão operacionalizadas no protocolo experimental.
Qualquer alteração posterior que mude uma hipótese, variável principal ou
critério de decisão deverá produzir uma nova versão documentada.

## 2. Unidade de análise e notação

- `u`: usuário real, participante autorizado ou usuário sintético explicitamente identificado;
- `i`: item do catálogo, representado por um filme;
- `p_u`: perfil declarado inicialmente pelo usuário;
- `h_u(t)`: histórico disponível para o usuário antes do instante `t`;
- `f_t`: feedback explícito positivo ou negativo registrado no instante `t`;
- `R_m(u,t,K)`: ranking Top-K produzido pelo método `m` usando somente `p_u` e `h_u(t)`;
- `y(u,i)`: relevância observada ou construída para o par usuário–item, com sua origem registrada;
- `K`: tamanho da lista de recomendação avaliada.

O perfil declarado `p_u` poderá conter até três gêneros ordenados, preferência
de década e preferência de popularidade. O histórico `h_u(t)` poderá conter
filmes apresentados, posição no ranking, score, feedback, contexto da busca e
versão do método. Informação posterior a `t` não pertence a `h_u(t)`.

A unidade principal de análise será o usuário ou perfil, conforme a origem dos
dados. Recomendações individuais de um mesmo usuário não serão tratadas como
observações independentes quando essa premissa não for válida.

## 3. Fontes de relevância

Toda observação usada na avaliação deverá declarar uma das seguintes fontes:

| Fonte | Definição | Uso permitido |
|---|---|---|
| Feedback real | Avaliação explícita fornecida por participante autorizado | Evidência principal quando houver amostra suficiente |
| Julgamento humano | Avaliação coletada em protocolo específico | Evidência de relevância, com acordo e procedimento documentados |
| Dado público | Avaliação obtida de dataset público autorizado | Avaliação offline conforme a licença e o protocolo |
| Usuário sintético | Preferência produzida por simulação controlada | Teste de comportamento, robustez e reprodutibilidade |
| Proxy heurístico | Rótulo calculado por regras do próprio sistema | Treino inicial e diagnóstico, sem equivalência com satisfação real |

Resultados de fontes diferentes serão apresentados separadamente. Uma métrica
calculada contra o proxy heurístico mede aderência ao proxy e não comprova a
satisfação de usuários.

## 4. Perguntas de pesquisa

### RQ1 — Efeito do feedback incremental

Em um cenário de cold start, qual é o impacto da incorporação de feedback
explícito incremental sobre a qualidade do ranking de um sistema de recomendação
baseado em preferências declaradas?

**Comparação principal:** método incremental completo versus o mesmo método com
perfil inicial estático, sob os mesmos usuários, candidatos e instantes.

**Resultados mínimos:** NDCG@K como métrica primária; Precision@K, Recall@K e
MRR como métricas complementares; intervalo de confiança, teste pareado e tamanho
de efeito; identificação da fonte de relevância.

### RQ2 — Quantidade de feedback

A partir de quantas interações o feedback produz mudança mensurável e estável na
qualidade do ranking?

**Comparação principal:** condições com 0, 1, 3, 5 e 10 interações, além do
histórico completo quando disponível.

**Resultados mínimos:** curva de desempenho por quantidade de interações,
incerteza por ponto, ganho marginal e análise de estabilidade.

### RQ3 — Relevância versus descoberta

A personalização incremental melhora a relevância sem degradar de forma
substancial a diversidade, a novidade e a cobertura do catálogo?

**Comparação principal:** perfil estático versus personalização incremental nas
mesmas condições de avaliação.

**Resultados mínimos:** NDCG@K em conjunto com diversidade intra-lista,
novidade, cobertura e viés de popularidade. Os resultados devem expor o
trade-off, não condensá-lo em uma única métrica sem justificativa.

### RQ4 — Desempenho, complexidade e custo operacional

Qual representação e método de recomendação oferecem a melhor relação entre
qualidade do ranking, complexidade e custo operacional?

**Comparação principal:** popularidade, content-based, TF-IDF com similaridade
de cosseno, modelo supervisionado, perfil estático e método incremental.

**Resultados mínimos:** métricas de ranking, tempo de treinamento quando
aplicável, latência de inferência, tamanho do artefato e consumo de memória sob
ambiente controlado.

### RQ5 — Contribuição dos componentes

Quais componentes da abordagem incremental são responsáveis pelos ganhos ou
perdas observados?

**Comparação principal:** sistema completo contra ablações sem feedback, sem
feedback negativo, sem gênero, sem diretor, sem filtros contextuais e sem
atributos textuais.

**Resultados mínimos:** diferença pareada por ablação, intervalo de confiança,
tamanho de efeito e análise dos cenários em que a remoção altera o resultado.

## 5. Hipóteses

As hipóteses abaixo são direcionais, mas não pressupõem que os resultados as
confirmarão. Para fins de teste, cada uma deve ser acompanhada da hipótese nula
de ausência de diferença relevante entre as condições comparadas.

### H1 — Benefício do feedback

O método incremental completo apresenta NDCG@K superior ao método de perfil
estático após incorporar feedback positivo e negativo.

**Variável independente:** uso de feedback incremental.
**Variável dependente primária:** NDCG@K.
**Controle principal:** mesmo usuário, instante, conjunto candidato, `K`, seed e
perfil inicial.
**Critério de decisão:** H1 somente poderá ser sustentada se a diferença for
positiva, o intervalo de confiança definido no protocolo for compatível com
ganho, o teste pareado atender ao limiar previamente definido e o tamanho de
efeito for reportado. Divergências em métricas secundárias serão discutidas.

### H2 — Aprendizado progressivo

A qualidade do ranking melhora à medida que aumenta a quantidade de interações
disponíveis, até atingir estabilização ou retorno marginal reduzido.

**Variável independente:** quantidade de feedback em `h_u(t)`.
**Variáveis dependentes:** NDCG@K, ganho marginal e estabilidade.
**Controle principal:** método incremental, perfil, catálogo, sequência de
eventos e seed.
**Critério de decisão:** H2 somente poderá ser sustentada se a tendência definida
no protocolo for positiva e não depender exclusivamente de um único ponto ou
subgrupo. A primeira quantidade com ganho mensurável deverá ser reportada.

### H3 — Mitigação do cold start

Preferências declaradas produzem rankings mais relevantes do que uma condição
sem informação pessoal durante o cold start.

**Variável independente:** quantidade de informação declarada no perfil.
**Variáveis dependentes:** NDCG@K, Precision@K, Recall@K e MRR.
**Comparação principal:** ausência de perfil versus 1 gênero, 3 gêneros, inclusão
de década e inclusão de popularidade.
**Critério de decisão:** H3 somente poderá ser sustentada se pelo menos a condição
de perfil completo superar a condição sem informação na métrica primária sob o
procedimento estatístico definido. Resultados intermediários deverão ser
reportados mesmo sem monotonicidade.

### H4 — Personalização versus popularidade

O método incremental apresenta maior relevância individual do que o baseline de
popularidade compatível com os mesmos filtros.

**Variável independente:** método de recomendação.
**Variáveis dependentes:** NDCG@K e MRR, acompanhados por novidade e viés de
popularidade.
**Controle principal:** usuário, candidatos, filtros, `K` e instante.
**Critério de decisão:** H4 somente poderá ser sustentada se o método incremental
superar o baseline na métrica primária pelo procedimento definido, sem ocultar
eventual piora nas métricas de descoberta.

### H5 — Trade-off entre relevância e diversidade

O ganho de relevância da personalização incremental pode ocorrer sem uma redução
substancial da diversidade das listas.

**Variável independente:** uso de personalização incremental.
**Variáveis dependentes:** NDCG@K e diversidade intra-lista; novidade, cobertura
e viés de popularidade como suporte.
**Controle principal:** usuário, candidatos, `K`, perfil, histórico e seed.
**Critério de decisão:** H5 somente poderá ser sustentada se houver ganho de
relevância e a variação de diversidade permanecer dentro da margem de não
inferioridade previamente fixada no protocolo. Essa margem ainda depende de
revisão metodológica.

## 6. Variáveis experimentais

### 6.1 Variáveis independentes

| Variável | Níveis planejados |
|---|---|
| Método | popularidade; content-based; TF-IDF + cosseno; supervisionado; perfil estático; incremental; híbrido, se justificado |
| Quantidade de feedback | 0; 1; 3; 5; 10; histórico completo |
| Tipo de feedback | nenhum; somente positivo; positivo e negativo; contraditório em robustez |
| Informação inicial | nenhuma; 1 gênero; 3 gêneros; + década; + popularidade; + feedback |
| Componentes ativos | configuração completa e ablações definidas em RQ5 |
| Fonte de relevância | real; julgamento humano; pública; sintética; heurística |

### 6.2 Variáveis dependentes

| Dimensão | Métricas | Papel |
|---|---|---|
| Relevância e ordenação | NDCG@K | Métrica primária de RQ1–RQ4 |
| Recuperação | Precision@K; Recall@K | Métricas complementares |
| Primeiro acerto | MRR | Métrica complementar de ordenação |
| Diversidade | diversidade intra-lista | Métrica primária do trade-off em H5 |
| Descoberta | novidade; cobertura; viés de popularidade | Métricas secundárias de RQ3 e H4 |
| Estabilidade | variação entre rankings; sensibilidade a feedback | Métricas de RQ2 e robustez |
| Eficiência | latência; tempo de treino; memória; tamanho do artefato | Métricas de RQ4 |

F1@K poderá ser reportada como síntese entre Precision@K e Recall@K, mas não
substituirá a métrica primária de ordenação.

### 6.3 Variáveis de controle

- versão e composição do catálogo;
- conjunto candidato disponível em cada instante;
- critérios de elegibilidade dos filmes;
- versão do dataset e origem dos rótulos;
- seed e estratégia de particionamento;
- valor de `K`;
- sequência e ordem temporal do feedback;
- perfil inicial e filtros contextuais;
- itens já apresentados ao usuário;
- versão do método, features, hiperparâmetros e threshold;
- hardware, versões das dependências e configuração de execução;
- política de desempate do ranking.

## 7. Matriz de rastreabilidade

| RQ | Hipótese | Comparação/experimento | Métrica principal | Métricas complementares | Evidência mínima |
|---|---|---|---|---|---|
| RQ1 | H1 | perfil estático × incremental | NDCG@K | Precision@K, Recall@K, MRR | comparação pareada, IC e tamanho de efeito |
| RQ2 | H2 | 0 × 1 × 3 × 5 × 10 × completo | NDCG@K | ganho marginal, estabilidade | curva por interação com incerteza |
| RQ3 | H5 | estático × incremental | NDCG@K + diversidade | novidade, cobertura, viés de popularidade | análise conjunta do trade-off |
| RQ4 | H3 | P0 × P1 × P2 × P3 × P4 × P5 | NDCG@K | Precision@K, Recall@K, MRR | comparação por nível de informação |
| RQ4 | H4 | popularidade × incremental | NDCG@K | MRR, novidade, viés de popularidade | comparação pareada e custo operacional |
| RQ4 | — | todos os métodos | NDCG@K | latência, memória, artefato, treino | ranking de qualidade e custo |
| RQ5 | H1, H5 | completo × cada ablação | NDCG@K | diversidade e métricas afetadas | diferença, IC e tamanho de efeito por ablação |

## 8. Diferenças entre os tipos de resultado

### 8.1 Classificação

Accuracy, Precision, Recall, F1, ROC-AUC e Average Precision calculadas sobre
linhas do dataset medem a capacidade de predizer a classe de aderência definida
para essas linhas. Quando o alvo é heurístico, elas medem reprodução do proxy.

### 8.2 Ranking

As métricas Top-K medem se itens relevantes aparecem e estão bem posicionados em
uma lista ordenada. Elas constituem a evidência principal para as perguntas deste
estudo, pois o sistema entrega rankings aos usuários.

### 8.3 Satisfação do usuário

Satisfação é um construto humano que não pode ser inferido somente de métricas de
classificação ou ranking. Caso seja avaliada, exigirá instrumento, consentimento
e protocolo próprios. Feedback binário é um sinal de preferência, mas não uma
medida completa de satisfação.

## 9. Ameaças preliminares à validade

### Validade interna

- vazamento de itens, usuários, contextos ou eventos futuros;
- seleção de threshold ou hiperparâmetros usando o holdout;
- dependência entre recomendações repetidas do mesmo usuário;
- rótulos construídos a partir das mesmas features usadas pelo modelo;
- exposição seletiva: o usuário somente avalia itens escolhidos pelo sistema.

### Validade de construto

- feedback binário pode não representar preferência estável;
- aderência heurística não equivale a relevância percebida;
- popularidade do TMDB pode não representar popularidade no público estudado;
- diversidade baseada em metadados pode não refletir diversidade percebida.

### Validade externa

- resultados no domínio de filmes podem não generalizar para outros domínios;
- catálogo filtrado por nota, votos, idioma e disponibilidade pode limitar a
  generalização;
- amostra pequena ou homogênea de usuários pode restringir as conclusões.

### Validade de conclusão

- baixa quantidade de usuários ou feedback reduz poder estatístico;
- múltiplas comparações aumentam o risco de falso positivo;
- médias agregadas podem ocultar subgrupos e usuários prejudicados;
- ausência de tamanho de efeito pode supervalorizar diferenças pequenas.

### Reprodutibilidade e operação

- mudanças no catálogo externo ao longo do tempo;
- versões não congeladas de bibliotecas e modelos;
- estado em memória e escrita concorrente do feedback em CSV;
- indisponibilidade de artefatos restritos por propriedade intelectual.

## 10. Resultados mínimos por pergunta

Uma pergunta será considerada respondida somente quando possuir:

1. comparação executada pelo mesmo protocolo e conjunto elegível;
2. métrica primária e métricas complementares previstas;
3. resultado por usuário ou perfil, além do agregado;
4. incerteza, teste aplicável e tamanho de efeito;
5. identificação da origem da relevância;
6. descrição de dados ausentes, falhas e exclusões;
7. discussão de limitações e resultados contrários à hipótese.

RQ1 e H1 são prioritárias. Se os dados forem insuficientes para responder às
demais perguntas com rigor, elas deverão ser reportadas como exploratórias ou
adiadas, sem ampliar artificialmente as conclusões.

## 11. Decisões pendentes

| ID | Decisão | Impacto | Responsável sugerido | Estado |
|---|---|---|---|---|
| DP-01 | Definir a fonte principal de relevância da avaliação final | validade das RQs e poder de generalização | autor e orientador | pendente |
| DP-02 | Definir os valores oficiais de `K` | comparabilidade das métricas | protocolo experimental | pendente |
| DP-03 | Definir limiar de significância, correção para múltiplos testes e método de IC | critérios das hipóteses | revisão metodológica | pendente |
| DP-04 | Definir margem de não inferioridade de diversidade para H5 | decisão sobre o trade-off | revisão metodológica | pendente |
| DP-05 | Definir tamanho mínimo da amostra e análise de poder | força das conclusões | autor e orientador | pendente |
| DP-06 | Definir se usuários sintéticos serão usados apenas em robustez ou também no benchmark | interpretação da evidência | protocolo experimental | pendente |
| DP-07 | Confirmar autorização institucional, autoria e artefatos publicáveis | submissão e reprodutibilidade externa | responsáveis institucionais | pendente |
| DP-08 | Confirmar evento, track e requisitos oficiais vigentes | formato e posicionamento do estudo | autores | pendente |

## 12. Revisão

Este documento está pronto para revisão, mas ainda não foi homologado pelo
orientador. A homologação ou as solicitações de alteração deverão ser registradas
no pull request ou em nova revisão versionada.
