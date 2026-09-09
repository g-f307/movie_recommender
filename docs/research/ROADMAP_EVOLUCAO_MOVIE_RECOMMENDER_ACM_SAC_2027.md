# Proposta de Evolução Científica do Movie Recommender
## Transformação do projeto AX Academy em pesquisa experimental para submissão ao ACM SAC 2027

**Versão:** 1.0
**Data:** 26 de agosto de 2026
**Projeto de origem:** AX Academy — Digital Transformation
**Projeto técnico:** Movie Recommender Bots
**Evento-alvo:** ACM Symposium on Applied Computing (SAC 2027)
**Track prioritário:** SEAI — Smarter Engineering: Building AI and Building with AI
**Prazo-alvo de submissão:** 2 de outubro de 2026 (EST)

---

## Estado da formalização científica

As perguntas de pesquisa, hipóteses, variáveis, critérios de decisão, ameaças
preliminares à validade e a matriz de rastreabilidade são formalizadas em
`docs/research/research_questions.md`.

O documento está em estado de candidato à revisão do orientador. Valores finais
de `K`, fonte principal de relevância, procedimento estatístico, tamanho mínimo
da amostra e margem de não inferioridade de diversidade permanecem como decisões
pendentes do protocolo experimental.

---

## 1. Visão geral

Esta proposta estabelece o plano de evolução do projeto **Movie Recommender Bots**, desenvolvido no contexto do **AX Academy — Digital Transformation**, para transformá-lo de uma aplicação de Machine Learning em um **estudo experimental reproduzível sobre personalização incremental em sistemas de recomendação conversacionais**.

O objetivo não é simplesmente produzir um novo sistema de recomendação de filmes. O objetivo é utilizar o sistema como **veículo experimental** para investigar uma questão científica relevante:

> **Em um cenário de cold start, a combinação entre preferências declaradas pelo usuário e feedback incremental melhora a qualidade, a relevância e a diversidade das recomendações quando comparada a abordagens baseadas exclusivamente no perfil inicial?**

A proposta preserva o trabalho já realizado e adiciona uma camada científica composta por:

- definição formal do problema;
- hipóteses testáveis;
- baselines;
- desenho experimental;
- métricas de ranking;
- ablações;
- análise estatística;
- avaliação do cold start;
- avaliação do feedback incremental;
- análise de diversidade, novidade e cobertura;
- reprodutibilidade;
- documentação de decisões de engenharia;
- preparação de artigo científico.

A proposta é especialmente alinhada ao **SEAI do ACM SAC 2027**, cujo escopo contempla engenharia de sistemas habilitados por IA, validação, monitoramento, arquitetura, human-in-the-loop, estudos empíricos, ferramentas, métodos e sistemas implantados.

---

## 2. Contexto e ponto de partida

O repositório atual já apresenta uma arquitetura funcional que integra aquisição e curadoria de dados, preparação de dataset, treinamento supervisionado, experiment tracking, API e interface conversacional.

A implementação atual contempla:

1. coleta de filmes a partir do TMDB;
2. curadoria do catálogo;
3. geração de dataset supervisionado;
4. comparação entre Logistic Regression, Linear SVC calibrado e Complement Naive Bayes;
5. validação cruzada por grupos;
6. holdout final;
7. seleção de modelo;
8. registro de experimentos com MLflow;
9. exposição do modelo por FastAPI;
10. interação via Telegram;
11. armazenamento de feedback por usuário;
12. personalização baseada no histórico;
13. controle de dados com DVC;
14. possibilidade de monitoramento de drift com Evidently.

A implementação atual já constitui uma base de engenharia considerável. O principal gap para uma publicação de conferência de alto nível não é a ausência de software, mas a ausência de uma **pergunta científica claramente isolada e demonstrada experimentalmente**.

---

# 3. Objetivo geral

Evoluir o Movie Recommender para uma plataforma experimental capaz de avaliar, de forma controlada e reproduzível, o impacto do **feedback incremental do usuário** sobre a qualidade das recomendações em um cenário de **cold start**.

---

# 4. Objetivos específicos

### O1 — Formalizar o problema

Definir matematicamente e operacionalmente:

- perfil inicial do usuário;
- catálogo de itens;
- representação dos filmes;
- função de aderência;
- mecanismo de ranking;
- feedback;
- atualização do perfil;
- condição de cold start.

### O2 — Estabelecer baselines

Implementar métodos de referência suficientemente fortes para determinar se a abordagem proposta realmente produz ganho.

### O3 — Avaliar a abordagem atual

Reproduzir os resultados do sistema existente sob protocolo experimental controlado.

### O4 — Isolar o efeito do feedback

Comparar recomendações sem histórico, com histórico e com diferentes quantidades de feedback.

### O5 — Avaliar qualidade além de acurácia

Medir simultaneamente:

- relevância;
- precisão do ranking;
- cobertura;
- diversidade;
- novidade;
- estabilidade;
- eficiência.

### O6 — Garantir reprodutibilidade

Versionar:

- dados;
- código;
- configuração;
- modelos;
- experimentos;
- métricas;
- seeds;
- ambiente.

### O7 — Produzir evidência científica

Gerar resultados suficientes para sustentar ou rejeitar as hipóteses definidas.

### O8 — Produzir o artigo

Desenvolver o paper em paralelo aos experimentos, evitando deixar a redação para o final.

---

# 5. Posicionamento científico

## 5.1 O que não será defendido

O projeto não deve ser apresentado como:

> "Um chatbot que recomenda filmes usando Machine Learning."

Essa formulação descreve a implementação, mas não apresenta uma contribuição científica forte.

## 5.2 O que será investigado

O trabalho deve ser enquadrado como um estudo experimental de:

> **personalização incremental em sistemas de recomendação conversacionais sob cold start.**

O sistema de filmes é o ambiente experimental no qual a hipótese é testada.

---

# 6. Pergunta de pesquisa

### RQ principal

**RQ1 — Em um cenário de cold start, qual é o impacto da incorporação de feedback incremental do usuário na qualidade do ranking produzido por um sistema de recomendação baseado em perfil declarado?**

### RQs secundárias

**RQ2 — Quanto feedback é necessário para produzir uma melhoria mensurável?**

**RQ3 — A personalização incremental melhora a relevância sem degradar diversidade, novidade e cobertura?**

**RQ4 — Qual representação e algoritmo apresentam melhor relação entre desempenho, complexidade e custo operacional?**

**RQ5 — Quais componentes da arquitetura são responsáveis pelos ganhos observados?**

---

# 7. Hipóteses

## H1 — Benefício do feedback

A incorporação de feedback positivo e negativo melhora a qualidade do ranking em relação ao modelo que utiliza apenas as preferências declaradas.

## H2 — Aprendizado progressivo

O desempenho melhora progressivamente à medida que aumenta a quantidade de interações disponíveis para o usuário.

## H3 — Cold start

A utilização de preferências declaradas permite obter desempenho competitivo mesmo na ausência de histórico prévio.

## H4 — Personalização versus popularidade

A personalização incremental apresenta maior relevância individual do que uma estratégia baseada exclusivamente em popularidade.

## H5 — Trade-off entre relevância e diversidade

O ganho de relevância obtido pela personalização não precisa resultar em concentração excessiva das recomendações.

---

# 8. Modelo conceitual

O sistema deverá ser analisado como um ciclo:

```text
                 ┌──────────────────────┐
                 │ Perfil declarado     │
                 │ 3 gêneros + filtros  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Cold-start model     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Ranking Top-N        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Interação do usuário │
                 │ Gostei / Não gostei  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Perfil incremental   │
                 └──────────┬───────────┘
                            │
                            └───────────────► novo ranking
```

O elemento central da pesquisa é o **loop de feedback**.

---

# 9. Arquitetura científica proposta

A arquitetura técnica deve ser organizada em camadas claramente separadas.

```text
┌─────────────────────────────────────────────────────────┐
│                  User Interaction Layer                 │
│                       Telegram                          │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Preference Layer                       │
│       Perfil declarado + histórico + feedback           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Recommendation Layer                   │
│ Baselines | ML | Hybrid | Incremental Personalization │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    Ranking Layer                         │
│       Score → Top-K → Diversificação opcional           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Evaluation Layer                       │
│ Relevance | Diversity | Novelty | Coverage | Stability │
└─────────────────────────────────────────────────────────┘
```

---

# 10. Evolução do dataset

## 10.1 Problema atual

O dataset atual combina supervisão heurística e feedback real. Essa característica precisa ser explicitamente controlada no experimento.

O artigo deve diferenciar:

- rótulo heurístico;
- rótulo derivado de feedback;
- dado observado;
- dado inferido;
- dado sintético, caso seja utilizado.

## 10.2 Nova estrutura mínima

Cada interação deverá possuir, quando possível:

```text
user_id
timestamp
session_id
movie_id
profile_genres
genre_rank_1
genre_rank_2
genre_rank_3
decade_preference
popularity_preference
model_version
recommendation_rank
recommendation_score
feedback
feedback_timestamp
```

## 10.3 Proibição de vazamento

O protocolo experimental deve impedir que informações futuras sejam utilizadas para produzir recomendações passadas.

O histórico disponível no instante `t` deve ser o único histórico utilizado para gerar a recomendação em `t+1`.

---

# 11. Baselines

Antes de comparar modelos sofisticados, devem ser estabelecidos baselines.

## B0 — Popularidade

Recomendar os itens mais populares compatíveis com os filtros.

Serve como baseline simples e forte.

## B1 — Content-Based

Representar filmes utilizando:

- sinopse;
- gêneros;
- diretor;
- demais metadados.

Calcular similaridade entre perfil e item.

## B2 — TF-IDF + Cosine Similarity

Baseline textual explícito e reproduzível.

## B3 — Modelo supervisionado atual

Comparar:

- Logistic Regression;
- Linear SVC calibrado;
- Complement Naive Bayes.

## B4 — Modelo personalizado sem atualização

O modelo recebe o perfil inicial, mas ignora feedback posterior.

## B5 — Modelo incremental

O modelo recebe:

- perfil inicial;
- feedback positivo;
- feedback negativo;
- histórico de itens;
- gêneros;
- diretores;
- palavras-chave.

## B6 — Abordagem proposta

Modelo híbrido final, caso os experimentos demonstrem vantagem.

---

# 12. Estratégia experimental

O experimento principal deverá simular diferentes níveis de conhecimento sobre o usuário.

## Condição C0 — Zero feedback

```text
Perfil declarado
        ↓
Recomendação
```

## Condição C1 — 1 interação

```text
Perfil declarado
        ↓
Recomendação
        ↓
1 feedback
        ↓
Nova recomendação
```

## Condição C2 — 3 interações

Avaliar o comportamento após três feedbacks.

## Condição C3 — 5 interações

Avaliar estabilização inicial.

## Condição C4 — 10 interações

Avaliar personalização após histórico mais substancial.

## Condição C5 — Histórico completo

Utilizar todo o histórico disponível, quando aplicável.

A comparação deverá permitir responder:

> **A partir de quantas interações o feedback começa a produzir ganho estatisticamente relevante?**

---

# 13. Avaliação offline

A avaliação offline deverá ser o núcleo quantitativo do artigo.

## Métricas de relevância

### Precision@K

Percentual dos itens recomendados no Top-K considerados relevantes.

### Recall@K

Proporção dos itens relevantes recuperados.

### F1@K

Equilíbrio entre Precision e Recall.

### NDCG@K

Avalia a qualidade da ordenação, dando maior peso às posições superiores.

### MRR

Avalia a posição do primeiro item relevante.

---

# 14. Métricas de diversidade e descoberta

Uma recomendação pode aumentar relevância e simultaneamente ficar repetitiva.

Por isso serão avaliadas:

### Intra-List Diversity

Mede a diversidade entre os itens de uma mesma lista.

### Catalog Coverage

Percentual do catálogo que consegue ser recomendado.

### Novelty

Mede a capacidade de recomendar itens menos óbvios para além dos títulos extremamente populares.

### Popularity Bias

Mede o quanto o sistema tende a concentrar recomendações nos itens populares.

---

# 15. Avaliação de estabilidade

O sistema deverá ser avaliado quanto à estabilidade do perfil.

Perguntas:

- O perfil converge?
- Um único feedback negativo distorce excessivamente as próximas recomendações?
- Feedbacks contraditórios são tratados adequadamente?
- O sistema retorna repetidamente aos mesmos filmes?
- O modelo apresenta comportamento instável após poucas interações?

---

# 16. Ablation Study

O artigo deve incluir um estudo de ablação.

Remover sistematicamente componentes da solução.

### A0 — Sistema completo

Todos os componentes.

### A1 — Sem feedback

Remove o histórico incremental.

### A2 — Sem feedback negativo

Utiliza apenas `Gostei`.

### A3 — Sem gênero

Remove a dimensão de gênero do perfil.

### A4 — Sem diretor

Remove informação de diretor.

### A5 — Sem filtros de década/popularidade

Remove filtros contextuais.

### A6 — Sem features textuais

Remove sinopse e demais features textuais.

O objetivo é responder:

> **Quais componentes realmente contribuem para o desempenho?**

---

# 17. Experimento de cold start

Este deverá ser um dos experimentos centrais do paper.

Criar perfis de usuários em diferentes níveis de informação:

| Perfil | Informação disponível |
|---|---|
| P0 | Nenhuma |
| P1 | 1 gênero |
| P2 | 3 gêneros |
| P3 | 3 gêneros + década |
| P4 | 3 gêneros + década + popularidade |
| P5 | Perfil completo + feedback |

Comparar a qualidade do ranking em cada condição.

Isso permite construir uma curva:

```text
Informação do usuário
        ↓
Qualidade da recomendação
```

---

# 18. Experimento de convergência

Para cada usuário:

1. gerar recomendação inicial;
2. registrar resultado;
3. aplicar feedback;
4. gerar nova recomendação;
5. repetir;
6. calcular métricas após cada interação.

O resultado esperado é uma curva do tipo:

```text
Qualidade
   ^
   |                 _________
   |             ___/
   |          __/
   |       __/
   |______/________________________> Interações
       0  1  2  3  5  10
```

A pergunta é se o sistema melhora rapidamente ou se o feedback apresenta pouco valor marginal.

---

# 19. Experimento de robustez

Testar:

- feedbacks aleatórios;
- feedbacks contraditórios;
- usuários com preferências muito específicas;
- usuários com preferências amplas;
- usuários com poucos itens compatíveis;
- gêneros pouco representados;
- filmes altamente populares;
- filmes pouco populares.

O objetivo é determinar em quais cenários o método é mais ou menos confiável.

---

# 20. Avaliação estatística

Os resultados não devem ser apresentados somente como médias.

Quando aplicável, utilizar:

- média;
- mediana;
- desvio padrão;
- intervalo de confiança;
- teste estatístico apropriado;
- tamanho de efeito.

A comparação entre métodos deve ser feita por usuário/perfil, evitando tratar cada recomendação individual como observação independente quando isso violar as premissas estatísticas.

---

# 21. Reprodutibilidade

A pesquisa deverá ser reproduzível por terceiros.

## Requisitos

Versionar:

- código;
- datasets permitidos;
- seeds;
- configurações;
- versões das bibliotecas;
- versões dos modelos;
- parâmetros;
- resultados;
- logs;
- ambiente de execução.

## Ferramentas existentes a preservar

- DVC;
- MLflow;
- FastAPI;
- Evidently.

## Adições recomendadas

- arquivo de configuração experimental;
- scripts de benchmark;
- scripts de avaliação;
- geração automática de tabelas;
- geração automática de gráficos;
- relatório final de experimentos.

---

# 22. Protocolo de execução

O projeto deverá possuir comandos equivalentes a:

```bash
# Preparação
python -m cinebot_ml.dataset

# Treinamento
python train_ml.py

# Benchmark
python experiments/run_benchmark.py

# Avaliação
python experiments/evaluate.py

# Ablation
python experiments/run_ablation.py

# Cold start
python experiments/run_cold_start.py

# Convergência
python experiments/run_convergence.py

# Geração dos resultados
python experiments/generate_tables.py
python experiments/generate_figures.py
```

Os nomes são propostos e podem ser adaptados à implementação final.

---

# 23. Estrutura recomendada do repositório

```text
movie_recommender/
│
├── cinebot_ml/
│   ├── dataset.py
│   ├── features.py
│   ├── models.py
│   ├── ranking.py
│   └── personalization.py
│
├── experiments/
│   ├── baselines/
│   ├── cold_start/
│   ├── convergence/
│   ├── ablation/
│   ├── robustness/
│   ├── benchmark.py
│   ├── evaluate.py
│   └── generate_results.py
│
├── evaluation/
│   ├── relevance.py
│   ├── diversity.py
│   ├── novelty.py
│   ├── coverage.py
│   └── statistics.py
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── reports/
│
├── notebooks/
│
├── docs/
│   ├── experimental_protocol.md
│   ├── data_dictionary.md
│   └── reproducibility.md
│
└── paper/
    ├── outline.md
    ├── figures/
    └── references.bib
```

---

# 24. Governança de dados e propriedade intelectual

Este ponto é obrigatório antes de qualquer submissão.

O repositório atual informa que o projeto, código, modelos e artefatos são propriedade da **LG Electronics do Brasil Ltda.** e que foi desenvolvido no âmbito do AX Academy / Convênio nº 005/2025 — INOVA / IFAM.

Portanto, antes de:

- publicar dataset;
- publicar código;
- publicar screenshots;
- descrever arquitetura proprietária;
- utilizar nome/logo da LG;
- mencionar informações internas;
- apresentar métricas obtidas em infraestrutura corporativa;
- divulgar resultados;

deve ser realizada uma revisão formal de publicação com os responsáveis pelo projeto e pela propriedade intelectual.

## Estratégia recomendada

Separar:

### Conteúdo público

- metodologia;
- algoritmos;
- métricas;
- resultados agregados;
- arquitetura abstrata;
- dataset público ou derivado de fonte pública;
- código especificamente autorizado.

### Conteúdo restrito

- credenciais;
- infraestrutura;
- dados internos;
- informações de usuários;
- componentes proprietários;
- detalhes estratégicos;
- informações comerciais.

---

# 25. Uso do contexto LG / AX Academy

A participação da LG deve aparecer como **contexto de aplicação e origem do problema**, não como propaganda.

Formulação recomendada:

> O estudo foi desenvolvido no contexto de um programa de formação e inovação aplicada em Inteligência Artificial, no qual desafios práticos foram utilizados como base para desenvolvimento e avaliação de sistemas inteligentes.

Quando autorizado, o artigo poderá reconhecer:

- LG Electronics;
- AX Academy;
- IFAM;
- equipes envolvidas;
- infraestrutura;
- orientação técnica.

A definição de autoria deve seguir contribuição científica efetiva e as regras do evento.

---

# 26. Contribuições esperadas do artigo

O paper deverá buscar sustentar de três a quatro contribuições claras.

### C1 — Método

Uma estratégia de personalização incremental baseada na combinação entre perfil declarado e feedback.

### C2 — Estudo experimental

Uma avaliação controlada do efeito do feedback em diferentes níveis de cold start.

### C3 — Evidência empírica

Resultados quantitativos sobre relevância, diversidade, novidade e cobertura.

### C4 — Artefato reproduzível

Uma implementação experimental versionada e documentada, quando a propriedade intelectual permitir.

---

# 27. Título provisório

### Opção principal

**Evaluating Incremental Personalization in Conversational Recommender Systems Under Cold-Start Conditions**

### Opção orientada à engenharia

**Engineering Incremental Personalization for Conversational Recommender Systems: An Empirical Study**

### Opção orientada à IA aplicada

**From Cold Start to Personalization: An Empirical Evaluation of Feedback-Driven Movie Recommendation**

O título definitivo deve ser escolhido somente após os resultados.

---

# 28. Estrutura prevista do artigo

## 1. Introduction

- problema;
- motivação;
- cold start;
- sistemas conversacionais;
- lacuna;
- pergunta de pesquisa;
- contribuições.

## 2. Background and Related Work

- recommender systems;
- content-based recommendation;
- conversational recommendation;
- cold start;
- incremental personalization;
- feedback loops.

## 3. Proposed Approach

- arquitetura;
- representação do usuário;
- representação dos itens;
- modelo;
- ranking;
- feedback;
- atualização do perfil.

## 4. Experimental Methodology

- dataset;
- baselines;
- protocolo;
- métricas;
- cenários;
- ablações;
- testes estatísticos.

## 5. Results

- desempenho geral;
- cold start;
- convergência;
- diversidade;
- novidade;
- cobertura;
- robustez.

## 6. Discussion

- interpretação;
- limitações;
- implicações de engenharia;
- implicações práticas.

## 7. Threats to Validity

- validade interna;
- validade externa;
- qualidade dos rótulos;
- viés do dataset;
- feedback implícito/explícito;
- limitações do domínio de filmes.

## 8. Conclusion

- resposta às RQs;
- principais resultados;
- trabalhos futuros.

---

# 29. Ameaças à validade

O artigo deverá tratar explicitamente:

## Validade interna

Possibilidade de vazamento de dados entre treino e teste.

## Validade externa

Resultados obtidos no domínio de filmes podem não generalizar diretamente para outros domínios.

## Qualidade da supervisão

Rótulos heurísticos podem introduzir viés.

## Viés de popularidade

Catálogos populares tendem a receber mais interações.

## Viés de feedback

Usuários podem fornecer feedback de forma inconsistente.

## Tamanho da amostra

Uma quantidade pequena de usuários/interações pode limitar a força estatística.

## Mudança temporal

Preferências e catálogo podem mudar ao longo do tempo.

---

# 30. Critérios de sucesso

A evolução será considerada cientificamente satisfatória se:

- [ ] existir uma pergunta de pesquisa clara;
- [ ] houver pelo menos 3 baselines relevantes;
- [ ] houver protocolo de cold start;
- [ ] houver avaliação incremental;
- [ ] houver métricas de ranking;
- [ ] houver métricas de diversidade/novidade;
- [ ] houver estudo de ablação;
- [ ] não houver leakage entre treino e avaliação;
- [ ] resultados forem estatisticamente analisados;
- [ ] experimentos forem reproduzíveis;
- [ ] limitações forem explicitamente documentadas;
- [ ] resultados sustentarem ou rejeitarem as hipóteses;
- [ ] propriedade intelectual tiver sido revisada;
- [ ] artigo estiver alinhado ao CFP do track escolhido.

---

# 31. Roadmap de execução

## Fase 0 — Validação institucional
### 26–28 de agosto

**Objetivo:** garantir que a pesquisa pode ser realizada e publicada.

### Atividades

- validar autorização para publicação;
- identificar orientador/coautor técnico;
- confirmar o que pode ser divulgado;
- definir política de dados;
- confirmar evento-alvo;
- definir autoria preliminar.

### Entregável

`docs/publication_clearance.md`

---

# Fase 1 — Auditoria científica do projeto
### 26–30 de agosto

**Objetivo:** entender exatamente o estado atual.

### Atividades

- revisar código;
- reproduzir treinamento;
- reproduzir métricas existentes;
- revisar dataset;
- revisar geração de rótulos;
- identificar leakage;
- documentar pipeline;
- registrar limitações.

### Entregáveis

- baseline reproduzido;
- relatório de auditoria;
- lista de gaps;
- protocolo experimental v1.

---

# Fase 2 — Definição experimental
### 29 de agosto – 3 de setembro

**Objetivo:** congelar a metodologia antes de começar a comparar resultados.

### Atividades

- definir RQs;
- definir hipóteses;
- definir baselines;
- definir métricas;
- definir splits;
- definir cenários;
- definir testes estatísticos.

### Entregável

`docs/experimental_protocol.md`

Este documento deve ser tratado como a especificação oficial dos experimentos.

---

# Fase 3 — Baselines
### 1–7 de setembro

Implementar:

- popularidade;
- content-based;
- TF-IDF;
- modelo supervisionado atual.

### Critério de saída

Todos os baselines devem gerar resultados através do mesmo pipeline de avaliação.

---

# Fase 4 — Personalização incremental
### 5–12 de setembro

Implementar/refatorar:

- perfil inicial;
- feedback positivo;
- feedback negativo;
- atualização do perfil;
- ranking personalizado;
- histórico por usuário;
- controle temporal do histórico.

### Critério de saída

O sistema deve ser capaz de executar uma sequência:

```text
profile
→ recommendation
→ feedback
→ updated profile
→ recommendation
→ feedback
→ ...
```

---

# Fase 5 — Avaliação
### 10–18 de setembro

Executar:

- benchmark;
- cold start;
- convergência;
- ablation;
- robustez;
- diversidade;
- novidade;
- cobertura.

### Entregáveis

- dataset experimental;
- resultados brutos;
- tabelas;
- gráficos;
- logs.

---

# Fase 6 — Análise estatística
### 17–21 de setembro

### Atividades

- calcular intervalos de confiança;
- realizar testes estatísticos;
- calcular tamanho de efeito;
- comparar modelos;
- identificar resultados significativos;
- documentar resultados negativos.

### Regra

Nenhuma hipótese deve ser considerada confirmada apenas porque uma média é maior.

---

# Fase 7 — Congelamento experimental
### 20–22 de setembro

Criar uma versão imutável do experimento:

```text
experiment-v1.0
```

Depois desse ponto:

- mudanças devem ser justificadas;
- resultados principais não devem ser manipulados;
- novas experiências devem ser registradas separadamente.

---

# Fase 8 — Escrita do paper
### 15–27 de setembro

A redação deve ocorrer paralelamente aos experimentos.

### Prioridade

1. metodologia;
2. resultados;
3. discussão;
4. introdução;
5. related work;
6. conclusão.

---

# Fase 9 — Revisão interna
### 25–29 de setembro

Revisão por:

- autor;
- orientador;
- especialista em ML;
- especialista em engenharia de software;
- responsável pelo projeto AX Academy;
- responsável por propriedade intelectual/publicação.

### Checklist

- [ ] metodologia consistente;
- [ ] números reproduzíveis;
- [ ] tabelas corretas;
- [ ] referências atualizadas;
- [ ] ausência de informação confidencial;
- [ ] autoria validada;
- [ ] formato ACM;
- [ ] inglês revisado.

---

# Fase 10 — Submissão
### 29 de setembro – 2 de outubro

### Entregáveis

- paper final;
- supplementary material, se permitido;
- artefatos autorizados;
- dados/metadados autorizados;
- informações de autores.

### Prazo-alvo

**2 de outubro de 2026 (EST).**

Não planejar submissão para as últimas horas.

---

# 32. Cronograma resumido

| Período | Fase | Resultado |
|---|---|---|
| 26–28/08 | Institucional | Autorização e autoria |
| 26–30/08 | Auditoria | Estado atual reproduzido |
| 29/08–03/09 | Metodologia | Protocolo congelado |
| 01–07/09 | Baselines | Referências implementadas |
| 05–12/09 | Personalização | Método incremental |
| 10–18/09 | Experimentos | Resultados |
| 17–21/09 | Estatística | Evidência |
| 20–22/09 | Freeze | Experimento final |
| 15–27/09 | Paper | Manuscrito |
| 25–29/09 | Revisão | Versão candidata |
| 29/09–02/10 | Submissão | Paper submetido |

---

# 33. Gestão de riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| Autorização de publicação não concedida | Muito alto | Resolver na Fase 0 |
| Dataset insuficiente | Alto | Criar protocolo controlado e/ou ampliar coleta autorizada |
| Feedback insuficiente | Alto | Avaliação offline e simulação controlada claramente identificada |
| Resultados sem ganho | Médio | Tratar resultado negativo como evidência científica |
| Leakage | Muito alto | Auditoria e splits temporais/grupais |
| Baixa significância estatística | Alto | Aumentar amostra e utilizar métricas por usuário |
| Escopo excessivo | Alto | Priorizar RQ1 e H1 |
| Falta de tempo | Muito alto | Congelar metodologia cedo |
| Problemas de propriedade intelectual | Muito alto | Revisão institucional antes da publicação |
| Mudança do CFP | Médio | Monitorar instruções oficiais do SAC |

---

# 34. Estratégia de priorização

Caso o tempo se torne insuficiente, a ordem de prioridade deverá ser:

### Prioridade P0

- autorização;
- RQ1;
- H1;
- baseline;
- modelo incremental;
- Precision@K;
- Recall@K;
- NDCG@K;
- cold start;
- análise estatística.

### Prioridade P1

- diversidade;
- novidade;
- cobertura;
- ablation study.

### Prioridade P2

- robustez;
- drift;
- otimizações arquiteturais;
- interface;
- experimentos adicionais.

A publicação não deve ser comprometida para implementar funcionalidades que não aumentem a evidência científica.

---

# 35. Resultado esperado

Ao final da evolução, o Movie Recommender deverá deixar de ser apresentado apenas como um projeto de aplicação e passar a constituir um **artefato experimental de pesquisa**.

O produto final será composto por:

```text
                 MOVIE RECOMMENDER
                         │
          ┌──────────────┼──────────────┐
          │              │              │
       Software       Dataset       Experimentos
          │              │              │
          └──────────────┼──────────────┘
                         │
                   Evidência
                         │
                         ▼
                    PAPER ACM
```

---

# 36. Critério final de decisão

A submissão ao ACM SAC 2027 deverá ocorrer somente se, após o congelamento experimental, houver:

1. uma contribuição claramente formulada;
2. comparação contra baselines;
3. evidência experimental suficiente;
4. análise estatística adequada;
5. discussão honesta das limitações;
6. alinhamento com o track;
7. autorização institucional e de propriedade intelectual.

Caso os resultados não demonstrem superioridade do método proposto, **não se deve fabricar uma narrativa de sucesso**. O artigo pode ser reposicionado como um estudo empírico sobre os limites da personalização incremental.

---

# 37. Enquadramento no ACM SAC 2027

O **SEAI — Smarter Engineering: Building AI and Building with AI** é o track prioritário desta proposta.

O escopo do track inclui explicitamente:

- engenharia do ciclo de vida de aplicações de IA;
- validação e monitoramento;
- arquitetura de sistemas baseados em IA;
- governança e rastreabilidade;
- human-in-the-loop;
- estudos empíricos sobre qualidade, confiabilidade e impacto de processos;
- ferramentas, métodos, datasets e benchmarks.

O Movie Recommender pode ser enquadrado principalmente como:

> **Empirical Study + AI-enabled System + Human-in-the-loop + Incremental Personalization**

Um segundo track potencial é **Machine Learning and Its Application (MLA)**, caso os resultados finais enfatizem mais o método de Machine Learning do que a engenharia do sistema.

---

# 38. Resultado institucional esperado para o AX Academy

Além da publicação científica, o projeto poderá produzir um conjunto de artefatos úteis ao programa:

- caso de aplicação de IA;
- benchmark de recomendação;
- metodologia experimental;
- pipeline reproduzível;
- estudo de personalização;
- documentação técnica;
- publicação internacional;
- apresentação em evento internacional;
- evidência de capacidade de desenvolvimento de IA aplicada.

O objetivo institucional é demonstrar que um desafio desenvolvido no AX Academy pode evoluir de:

```text
Desafio prático
      ↓
Protótipo
      ↓
Sistema funcional
      ↓
Experimento controlado
      ↓
Evidência científica
      ↓
Publicação internacional
```

---

# 39. Próximas ações imediatas

1. **Obter autorização formal para transformar o projeto em publicação.**
2. **Definir orientador/coautores e regras de autoria.**
3. **Reproduzir exatamente o estado atual do projeto.**
4. **Auditar o dataset e os rótulos.**
5. **Implementar o primeiro baseline não-ML.**
6. **Implementar a avaliação Top-K.**
7. **Implementar o protocolo de cold start.**
8. **Formalizar o mecanismo de feedback incremental.**
9. **Executar o primeiro benchmark.**
10. **Somente depois ampliar para ablação, diversidade e robustez.**

---

## 40. Decisão arquitetural recomendada

A principal mudança conceitual proposta é separar claramente:

```text
                    SISTEMA
                       │
        ┌──────────────┴──────────────┐
        │                             │
   PRODUÇÃO                       PESQUISA
        │                             │
 Telegram + API                Experimentos
        │                             │
        │                     Baselines
        │                     Ablations
        │                     Cold Start
        │                     Convergence
        │                     Statistics
        │                             │
        └──────────────┬──────────────┘
                       │
                 MESMO DATASET
                 MESMA DEFINIÇÃO
                 MESMO PROTOCOLO
```

O sistema de produção não deve ser confundido com o instrumento científico.

A aplicação serve para demonstrar a engenharia e permitir interações; o **pipeline experimental** será responsável por produzir a evidência que sustenta o artigo.

---

## 41. Visão de longo prazo

Mesmo que a submissão ao SAC 2027 não seja aceita, a evolução proposta cria uma base reutilizável para:

- artigos posteriores;
- trabalhos de conclusão;
- iniciação científica;
- novas aplicações de recomendação;
- sistemas conversacionais;
- personalização;
- agentes inteligentes;
- projetos futuros do AX Academy.

O investimento principal, portanto, não é somente a submissão de um artigo. É a transformação de um projeto aplicado em um **artefato de pesquisa reproduzível**, capaz de gerar novas hipóteses, experimentos e publicações.

---

## Referências de planejamento

- ACM Symposium on Applied Computing 2027 — página oficial.
- SAC 2027 — Technical Tracks.
- SEAI @ ACM SAC 2027 — Smarter Engineering: Building AI and Building with AI.
- Repositório Movie Recommender Bots — AX Academy / Digital Transformation.

**Observação:** datas, regras de submissão, formatação, limites de páginas, anonimização, autoria e requisitos de artefatos devem ser confirmados na versão oficial do CFP e das instruções de submissão no momento da preparação final.
