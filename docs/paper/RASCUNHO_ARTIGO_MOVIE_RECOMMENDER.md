# Personalização incremental em sistemas conversacionais de recomendação sob condições de cold start

## Resumo

Sistemas de recomendação conversacionais precisam produzir sugestões relevantes mesmo quando possuem pouca ou nenhuma informação histórica sobre o usuário. Esse problema, conhecido como cold start, é especialmente desafiador quando as preferências são declaradas de forma limitada e as interações posteriores fornecem sinais positivos e negativos de maneira incremental. Este trabalho propõe um estudo experimental sobre o impacto da incorporação de feedback explícito em um sistema conversacional de recomendação de filmes. A solução combina um perfil inicial, composto por três gêneros ordenados e filtros de década e popularidade, com informações textuais e estruturadas dos filmes, modelos supervisionados e uma camada de personalização incremental baseada no histórico do usuário. O sistema será avaliado por meio de baselines de popularidade, similaridade textual, recomendação content-based e modelos supervisionados, considerando diferentes quantidades de feedback. A avaliação contemplará métricas de ranking, como Precision@K, Recall@K, NDCG@K e MRR, além de diversidade, novidade, cobertura, estabilidade e eficiência. O objetivo é verificar se o feedback incremental melhora a relevância individual sem provocar concentração excessiva do ranking, bem como identificar a quantidade de interações necessária para produzir ganhos mensuráveis. Como resultado esperado, o trabalho pretende contribuir com uma avaliação controlada e reproduzível de personalização incremental em um sistema conversacional aplicado ao domínio de filmes.

**Palavras-chave:** sistemas de recomendação; cold start; personalização incremental; recomendação conversacional; feedback do usuário; Machine Learning.

## Introdução

Sistemas de recomendação são utilizados para reduzir o excesso de opções disponíveis em catálogos digitais e auxiliar usuários na descoberta de itens de interesse. Apesar dos avanços em filtragem colaborativa, modelos baseados em conteúdo e métodos híbridos, a qualidade das recomendações depende da disponibilidade de informações sobre os usuários e sobre os itens. Quando um novo usuário ainda não possui histórico de interações, ocorre o problema de cold start, que limita a capacidade de personalização do sistema.

Uma alternativa para reduzir esse problema consiste em solicitar preferências explícitas no início da interação. Em um sistema conversacional, essas preferências podem ser coletadas de forma gradual e combinadas com feedback posterior, como avaliações positivas e negativas. Dessa forma, o recomendador deixa de operar apenas com um perfil inicial estático e passa a atualizar sua representação do usuário ao longo da sessão ou de sessões subsequentes.

Entretanto, a existência de feedback não garante, por si só, melhoria nas recomendações. O feedback pode ser escasso, contraditório ou enviesado pelos próprios itens apresentados pelo sistema. Além disso, o aumento da relevância pode ocorrer às custas de diversidade, novidade e cobertura do catálogo. Assim, é necessário avaliar não apenas se o modelo classifica corretamente a aderência de um filme ao perfil, mas se produz rankings melhores em diferentes estágios de conhecimento sobre o usuário.

Este trabalho utiliza o Movie Recommender, desenvolvido no contexto do AX Academy — Digital Transformation, como plataforma experimental para investigar a seguinte questão:

> Em um cenário de cold start, a incorporação de feedback incremental melhora a qualidade do ranking produzido por um sistema conversacional baseado em preferências declaradas?

A proposta combina um catálogo de filmes obtido de fonte pública, representação textual e estruturada dos itens, modelos supervisionados, baselines não supervisionados e uma camada de personalização incremental. O usuário informa três gêneros em ordem de preferência, uma década e um perfil de popularidade. Após receber recomendações, pode indicar que gostou ou não gostou dos filmes apresentados. Essas interações são utilizadas para atualizar o perfil e produzir novas listas.

As principais contribuições esperadas são:

1. uma arquitetura de recomendação conversacional que combina perfil declarado e feedback incremental;
2. um protocolo experimental para avaliar a evolução da qualidade em diferentes níveis de cold start;
3. uma comparação entre baselines e modelos supervisionados usando métricas de ranking, diversidade, novidade e cobertura;
4. um artefato experimental versionado, com dados, configurações, modelos e resultados rastreáveis, respeitando as restrições de propriedade intelectual aplicáveis.

## Metodologia

### Desenho do sistema

O sistema será estruturado em quatro camadas principais. A camada de interação será responsável pela coleta das preferências e do feedback por meio de uma interface conversacional. A camada de perfil manterá as preferências declaradas e o histórico de interações. A camada de recomendação calculará a aderência dos itens utilizando baselines, modelos supervisionados e personalização incremental. Por fim, a camada de avaliação produzirá métricas, logs e artefatos experimentais.

O ciclo de recomendação será definido por:

```text
perfil declarado → ranking inicial → interação → atualização do perfil → novo ranking
```

O perfil inicial será composto por três gêneros ordenados, preferência de década e preferência de popularidade. O catálogo conterá atributos como sinopse, gêneros, diretor, ano, nota, quantidade de votos, serviços de streaming e palavras-chave.

### Representação e geração dos dados

Os filmes serão representados por uma combinação de atributos textuais, categóricos e numéricos. As sinopses, gêneros e perfil textual serão transformados por TF-IDF. Atributos categóricos, como gênero principal, diretor, década e preferência de popularidade, serão codificados por variáveis categóricas. Atributos numéricos, como ano, nota, votos, duração e quantidade de serviços de streaming, serão normalizados.

Durante a etapa inicial, poderão ser utilizados rótulos heurísticos de aderência para construir o modelo de cold start. Esses rótulos serão identificados separadamente dos rótulos derivados de feedback real. O estudo deverá reportar essa distinção e realizar análises de sensibilidade, pois um modelo treinado apenas sobre rótulos heurísticos mede principalmente sua capacidade de reproduzir o proxy definido.

Cada evento experimental deverá registrar, quando disponível, `user_id`, `session_id`, timestamp, `movie_id`, versão do modelo, posição no ranking, score, perfil utilizado, feedback e timestamp do feedback. O protocolo impedirá que feedback futuro seja utilizado na geração de recomendações anteriores.

### Métodos comparados

Serão avaliados os seguintes métodos:

- **B0 — Popularidade:** recomenda os filmes mais populares compatíveis com os filtros.
- **B1 — Content-based:** calcula similaridade entre o perfil do usuário e os atributos dos filmes.
- **B2 — TF-IDF + similaridade de cosseno:** baseline textual explícito e reproduzível.
- **B3 — Modelo supervisionado:** compara Logistic Regression, Linear SVC calibrado e Complement Naive Bayes.
- **B4 — Perfil estático:** utiliza as preferências iniciais e ignora feedback posterior.
- **B5 — Personalização incremental:** incorpora feedback positivo, feedback negativo e atributos dos filmes já avaliados.
- **B6 — Modelo híbrido:** combina o score do modelo supervisionado com o score de personalização e regras de diversificação, caso os experimentos demonstrem vantagem.

### Condições experimentais

O efeito da quantidade de informação disponível será avaliado nas seguintes condições:

- **C0:** nenhuma informação histórica;
- **C1:** perfil inicial e uma interação;
- **C2:** perfil inicial e três interações;
- **C3:** perfil inicial e cinco interações;
- **C4:** perfil inicial e dez interações;
- **C5:** perfil inicial e histórico completo disponível.

Para cada condição, o sistema deverá gerar uma recomendação, aplicar apenas o feedback disponível naquele momento e produzir o ranking seguinte. Esse procedimento permitirá construir uma curva de evolução da qualidade em função do número de interações.

### Avaliação

A relevância será avaliada por Precision@K, Recall@K, F1@K, NDCG@K e MRR. Quando não houver avaliação explícita suficiente, os resultados deverão distinguir claramente entre relevância observada, relevância heurística e relevância obtida por julgamento humano.

Também serão avaliadas:

- diversidade intra-lista;
- cobertura do catálogo;
- novidade dos itens recomendados;
- viés de popularidade;
- repetição de filmes;
- estabilidade após feedback positivo, negativo e contraditório;
- tempo e custo de inferência.

O estudo incluirá um experimento de ablação com as seguintes variantes: sistema completo, sem feedback, sem feedback negativo, sem gênero, sem diretor, sem filtros de década e popularidade e sem atributos textuais. Também serão avaliados cenários de robustez com usuários de preferências amplas ou específicas, gêneros pouco representados, poucos itens compatíveis e feedbacks aleatórios ou contraditórios.

As comparações serão realizadas por usuário e perfil, evitando tratar cada recomendação individual como observação independente quando isso violar as premissas estatísticas. Serão reportadas médias, medianas, desvios padrão, intervalos de confiança, testes estatísticos apropriados e tamanho de efeito.

### Reprodutibilidade e governança

Os dados autorizados, o código experimental, as configurações, as seeds, as versões das bibliotecas, os modelos e os resultados serão versionados com DVC, MLflow e controle de versão. Scripts específicos deverão automatizar benchmark, avaliação, ablação, cold start, convergência, robustez, geração de tabelas e geração de figuras.

Como o projeto foi desenvolvido no contexto da LG Electronics do Brasil, do AX Academy e do IFAM, a publicação deverá passar por revisão institucional e de propriedade intelectual. Dados de usuários, credenciais, infraestrutura interna e componentes não autorizados não serão publicados.

## Resultados e discussões

Os resultados finais serão organizados em cinco análises complementares.

### Desempenho dos métodos

Inicialmente, serão comparados os baselines e os modelos supervisionados usando as métricas de ranking. Essa análise permitirá verificar se a solução proposta supera estratégias simples de popularidade e similaridade textual. As métricas de classificação do modelo supervisionado serão apresentadas apenas como diagnóstico do componente de aderência, não como substitutas da avaliação do ranking.

### Efeito do feedback incremental

A comparação entre C0 e as demais condições permitirá estimar o ganho associado ao conhecimento progressivo do usuário. A hipótese H1 será sustentada somente se a personalização incremental produzir melhoria consistente em relação ao perfil estático. A hipótese H2 será analisada observando-se se existe crescimento progressivo ou se o ganho se concentra nas primeiras interações.

É possível que o feedback produza ganhos pequenos, nulos ou negativos em alguns cenários. Esse resultado não invalidará o estudo; ele poderá indicar que o mecanismo de atualização é sensível à qualidade do feedback, à escassez de itens compatíveis ou ao viés do catálogo.

### Relevância, diversidade e descoberta

A análise deverá verificar se o aumento da relevância vem acompanhado de concentração excessiva em poucos gêneros, diretores ou filmes populares. Um método será considerado mais equilibrado quando melhorar a relevância sem reduzir substancialmente diversidade, novidade e cobertura.

### Contribuição dos componentes

O estudo de ablação mostrará quais atributos e mecanismos são responsáveis pelo desempenho. Por exemplo, uma queda significativa sem feedback indicará a importância da atualização do perfil; uma queda sem atributos textuais indicará contribuição da sinopse e das palavras-chave; uma redução pequena sem diretor poderá revelar que essa informação tem baixo valor no catálogo analisado.

### Interpretação e limitações

Os resultados deverão ser interpretados com cautela devido à possibilidade de rótulos heurísticos, amostra limitada de usuários, viés de popularidade e especificidade do domínio de filmes. Métricas muito altas obtidas sobre rótulos derivados da própria heurística não deverão ser apresentadas como prova de satisfação do usuário.

## Considerações finais

Este trabalho propõe transformar um sistema funcional de recomendação de filmes em uma plataforma experimental para investigar personalização incremental em ambientes conversacionais e de cold start. A abordagem parte de preferências declaradas e incorpora feedback explícito para atualizar o perfil do usuário e gerar novos rankings.

O ponto central da investigação não é apenas construir um chatbot, mas medir sob condições controladas se o feedback melhora a recomendação, quantas interações são necessárias para produzir ganho, quais componentes são responsáveis pelo comportamento observado e quais compromissos existem entre relevância, diversidade, novidade e cobertura.

Espera-se que o estudo contribua tanto para a engenharia de sistemas de recomendação quanto para a compreensão empírica dos limites da personalização incremental. Mesmo que a hipótese de melhoria não seja confirmada em todos os cenários, resultados negativos ou condicionais poderão revelar informações relevantes sobre qualidade dos dados, sensibilidade do mecanismo de atualização e adequação de diferentes estratégias em situações de cold start.

Como trabalhos futuros, poderão ser investigados modelos de atualização online, aprendizado de ranking, calibração de confiança, explicações personalizadas, maior diversidade de catálogos e avaliação com usuários em escala ampliada.

## Referências

BURKE, Robin. Hybrid recommender systems: survey and experiments. *User Modeling and User-Adapted Interaction*, v. 12, p. 331–370, 2002.

HERLOCKER, Jonathan L. et al. Evaluating collaborative filtering recommender systems. *ACM Transactions on Information Systems*, v. 22, n. 1, p. 5–53, 2004.

JANNACH, Dietmar et al. *Recommender Systems: An Introduction*. Cambridge: Cambridge University Press, 2010.

KOREN, Yehuda; BELL, Robert; VOLINSKY, Chris. Matrix factorization techniques for recommender systems. *Computer*, v. 42, n. 8, p. 30–37, 2009.

LOPS, Pasquale; DE GEMMIS, Marco; SEMERARO, Giovanni. Content-based recommender systems: state of the art and trends. In: RICCI, Francesco et al. (ed.). *Recommender Systems Handbook*. Boston: Springer, 2011. p. 73–105.

RICCI, Francesco; ROKACH, Lior; SHAPIRA, Bracha (ed.). *Recommender Systems Handbook*. 2. ed. Boston: Springer, 2015.

SCHEIN, Andrew I. et al. Methods and metrics for cold-start recommendations. In: *Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval*. New York: ACM, 2002. p. 253–260.

Nota: as referências específicas sobre recomendação conversacional, feedback incremental, diversidade e avaliação devem ser ampliadas durante a revisão bibliográfica e formatadas conforme o modelo oficial da ACM.
