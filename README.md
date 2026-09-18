# Movie Recommender Bots

> **Propriedade Intelectual:** Este projeto e todo o seu código-fonte, modelos e artefatos são de propriedade da LG Electronics do Brasil Ltda., desenvolvido no âmbito do projeto AX Academy --- Digital Transformation (Convênio N.º 005/2025 --- INOVA / IFAM). Consulte o arquivo LICENSE para mais detalhes.

Upgrade do `CineBot, curadoria de filmes` para o `Desafio 02 - Upgrade do Bot Pessoal com ML`. O projeto preserva os três bots originais do Módulo 1 e adiciona uma camada de Machine Learning para recomendar filmes a partir de `3 gêneros ranqueados pelo usuário`, mais filtros de `década` e `popularidade`, como indicado na Seção 8 do enunciado para `Gabriel de Sá`.

Nesta versão, o repositório também funciona como plataforma experimental para
avaliar personalização incremental em cold start. A evidência principal do
estudo é offline e sintética; feedback real permanece escasso e sempre é
identificado separadamente.

## Arquitetura

O fluxo está dividido entre aplicações operacionais e infraestrutura científica:

1. `gabriel-scrapper/`
   Coleta filmes do TMDB com BotCity Web, prioriza sinopses em `pt-BR`, limita excesso de franquias via coleção do TMDB, extrai diretor, gêneros, streaming e gera `data/filmes.json`.
2. `gabriel-curadoria/`
   Lê a base coletada e monta uma fila local em `data/fila_curadoria.json`. O envio ao DataPool do Maestro ficou opcional.
3. `cinebot_ml/`
   Implementa dados, splits, B0–B5, métricas, personalização, agentes sintéticos, replay C0–C5, benchmark e matriz experimental.
4. `configs/`
   Contém parâmetros científicos validados por JSON Schema e protegidos por locks SHA-256.
5. `docs/research/`
   Registra protocolo, contratos, decisões metodológicas e limitações.
6. `main.py`
   Sobe uma API FastAPI com `GET /saude` e `POST /predict`.
7. `gabriel-telegram/`
   Recebe `3 gêneros ranqueados`, `década` e `popularidade`, chama o endpoint de ML, obtém um ranking top-5, apresenta uma sugestão por vez, salva feedback por `user_id` e personaliza as próximas recomendações daquele usuário.

## Estrutura

```text
.
├── .dvc/
├── .github/workflows/       # validação automática
├── cinebot_ml/
│   ├── experiments/         # matriz, execução e retomada
│   ├── personalization/     # estado e atualização incremental
│   ├── ranking/             # contratos, B0–B5 e métricas
│   └── simulation/          # agentes e replay temporal
├── configs/
│   ├── agents/
│   ├── experiments/
│   └── methods/
├── data/                    # catálogo local, não versionado
├── datasets/                # datasets locais e ponteiros DVC
├── docs/
│   ├── paper/
│   └── research/
├── gabriel-curadoria/
├── gabriel-scrapper/
├── gabriel-telegram/
├── tests/
├── dvc.yaml
├── LICENSE
├── Makefile
├── main.py
├── README.md
├── requirements-ci.txt
├── requirements-ml.txt
└── train_ml.py
```

## Aplicação de ML escolhida

Mapeamento do desafio para este projeto:

- Aluno: `Gabriel de Sá`
- Projeto do Módulo 1: `CineBot, curadoria de filmes`
- Direção sugerida na Tabela 8.2: recomendação content-based sobre `sinopse + gêneros`
- Entrada do usuário: `3 gêneros ranqueados + década + popularidade`
- Saída do bot: `top-N de filmes mais aderentes`, ordenados pelo score do modelo

Para operacionalizar isso no desafio, a camada de ML usa:

- features textuais: sinopse, gêneros do filme e perfil textual das preferências
- features categóricas: gênero principal do filme, diretor, ranking dos gêneros escolhidos, década desejada e perfil de popularidade
- features numéricas: ano, nota, votos, duração, quantidade de streamings e aderência às preferências de década e popularidade
- target supervisionado: aderência do filme ao perfil ranqueado declarado
- personalização adicional: reforço por feedback real do usuário na inferência e, quando houver contexto compatível, sobrescrita do rótulo no dataset

## Modelos comparados

O script de treino foi preparado para comparar no MLflow:

- `LogisticRegression`
- `LinearSVC` calibrado
- `ComplementNB`

O vencedor é escolhido com base em `F1`, `Precision`, `PR AUC` e `ROC-AUC`, após validação cruzada por grupos (`movie_id`) no conjunto de treino, e depois é confirmado em um holdout final separado. O modelo vencedor é salvo localmente em `artifacts/production_model.joblib` e, quando o MLflow estiver configurado, também pode ser promovido no Registry com alias `@production`.

## Requisitos

Antes de rodar:

- Python `3.11–3.14` para a infraestrutura experimental;
- Python `3.11` e `3.12` são validados automaticamente na CI;
- BotCity Framework Web
- token do Telegram
- chave da API do TMDB
- Chromium ou Google Chrome
- ChromeDriver compatível
- conta no BotCity Maestro apenas se quiser manter o fluxo opcional com DataPool

Dependências:

- bots originais:
  - `gabriel-scrapper/requirements.txt`
  - `gabriel-curadoria/requirements.txt`
  - `gabriel-telegram/requirements.txt`
- camada de ML:
  - `requirements-ml.txt`

## Instalação experimental

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
make validate
make test
```

`make setup` é o comando único para instalar as dependências necessárias aos
contratos, baselines, simulação e testes. Para instalar também MLflow, DVC, API
e os três bots, use `make setup-full`.

O benchmark offline, os testes e o smoke test não exigem `.env`, token do
Telegram, credencial do Maestro ou chave do TMDB.

## Prontidão experimental

```bash
make validate   # Python, imports, schemas, locks e descritores de dados
make compile    # compilação dos módulos
make smoke      # B0–B5 em dados mínimos
make test       # suíte completa
make readiness  # inclui dados e artefatos locais da execução completa
```

A CI executa esse fluxo em Python 3.11 e 3.12. O modo `readiness` falha de forma
explícita quando catálogo, dataset, remoto DVC ou artefatos B2/B3 não estiverem
disponíveis e compatíveis.

## Matriz experimental

```bash
# enumera as 15.120 células oficiais sem executá-las
python -m cinebot_ml.experiments list

# exemplo de matriz mínima
python -m cinebot_ml.experiments list \
  --method B0 --condition C0 --profile P0 \
  --persona consistent --seed 42 --k 5
```

A execução recebe um runner específico do estudo por `--runner modulo:funcao` e
persiste manifesto, status e um checkpoint atômico por célula. Uma retomada
ignora resultados existentes e não os sobrescreve. Consulte
[`docs/research/experiment_matrix.md`](docs/research/experiment_matrix.md).

O estudo pareado de cold start pode ser executado por configuração, preservando
o mesmo agente latente entre P0–P5:

```bash
python -m cinebot_ml.experiments cold-start \
  --catalog data/filmes.json --condition C2 \
  --persona consistent --seed 42 --k 5 \
  --method B0 --method B1 --method B4 --method B5 \
  --output results/raw/cold_start
```

As saídas incluem dataset bruto, disponibilidade de candidatos, manifesto e
resumo descritivo. B2 e B3 exigem seus artefatos explícitos.

No estado atual não há remoto DVC compartilhado configurado. Assim, `dvc pull`
sozinho não reconstrói os ativos em um clone novo. O catálogo pode ser
regenerado pelo scraper com `TMDB_API_KEY`; dataset e B3 podem ser regenerados
pelos estágios do `dvc.yaml`. O artefato B2 deve ser ajustado exclusivamente na
partição de treino. Antes da execução oficial, esses ativos devem ser publicados
em armazenamento autorizado ou regenerados e congelados conforme o protocolo.

Com um manifesto de split aprovado, gere B2 somente com a partição de treino:

```bash
make prepare-b2 SPLIT_MANIFEST=results/manifests/splits/<split_id>.json
```

B3 é regenerado por `make prepare-b3`; o treinamento atual grava os campos de
rastreabilidade exigidos pelo contrato. Artefatos locais antigos sem esses
campos são rejeitados por `make readiness`.

## Configuração operacional dos bots

Somente as integrações operacionais precisam de `.env`. Crie-o na raiz quando
for executar scraper, Maestro, API ou Telegram:

```env
MAESTRO_SERVER=https://SEU_SERVIDOR_MAESTRO
MAESTRO_LOGIN=SEU_LOGIN
MAESTRO_KEY=SUA_CHAVE
TMDB_API_KEY=SUA_TMDB_API_KEY
TELEGRAM_BOT_TOKEN=SEU_TOKEN_LOCAL_OPCIONAL
CINEBOT_ML_API_URL=http://127.0.0.1:8000
DATA_PATH=data/filmes.json
MOVIES_PER_PROFILE=0
MAX_PAGES=8
MAX_MOVIES_PER_COLLECTION=1
MIN_VOTE_AVERAGE=7.2
MIN_VOTE_COUNT=300
DISCOVER_SORT_BY=popularity.desc
CINEBOT_ENABLE_DRIFT_ON_PREDICT=false
LOCAL_QUEUE_PATH=data/fila_curadoria.json
CURADORIA_HISTORY_PATH=data/historico_inseridos.json
USE_DATAPOOL=false
HEADLESS=true
```

Credenciais esperadas no Maestro:

- `gabriel-tmdb` / campo `api_key`
- `gabriel-telegram` / campo `token`

## Pipeline de dados

### 1. Coletar catálogo

```bash
python gabriel-scrapper/bot.py
```

Saída principal:

- `data/filmes.json`

### 2. Gerar fila de curadoria

```bash
python gabriel-curadoria/bot.py
```

Saída principal:

- `data/fila_curadoria.json`
- `data/historico_inseridos.json`

### 3. Gerar dataset versionado

```bash
python3 -m cinebot_ml.dataset
```

Arquivos gerados:

- `datasets/movie_preferences.csv`
- `datasets/user_feedback.csv`

O repositório já inclui:

- `dvc.yaml`
- `datasets/movie_preferences.csv.dvc`

Se o DVC estiver instalado, você pode rastrear novamente o dataset com os comandos usuais do DVC.

### 4. Treinar e promover o modelo

```bash
python3 train_ml.py
```

Artefatos esperados:

- `artifacts/production_model.joblib`
- `artifacts/model_metadata.json`
- `datasets/reference_features.csv`

Opcionalmente, para visualizar o Tracking:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

O treino atual inclui:

- comparação de `3` famílias de modelos distintas
- validação cruzada por grupos com `StratifiedGroupKFold` quando disponível
- holdout final com `GroupShuffleSplit`
- seleção de threshold por desempenho médio nas folds
- persistência das métricas médias e do desvio padrão em `artifacts/model_metadata.json`
- resumo formal da supervisão da base, incluindo proporção de rótulos heurísticos versus rótulos sobrescritos por feedback real

## FastAPI

Suba o serviço local:

```bash
uvicorn main:app --reload
```

Healthcheck:

```bash
curl http://127.0.0.1:8000/saude
```

Predição:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ranked_genres":["acao","drama","comedia"],"decade_preference":"moderno","popularity_preference":"joia_escondida","data_path":"data/filmes.json","top_n":5,"user_id":"12345"}'
```

## Bot do Telegram com ML

```bash
python gabriel-telegram/bot.py
```

Comandos disponíveis:

- `/start`
- `/help`
- `/recomendar`
- `/sugestao`

Fluxo do `/sugestao`:

1. o usuário escolhe o gênero favorito
2. escolhe o segundo gênero
3. escolhe o terceiro gênero
4. escolhe a década: `Moderno`, `Anos 2000` ou `Antes dos anos 2000`
5. escolhe a popularidade: `Popular` ou `Joia escondida`
6. o bot chama `POST /predict`
7. o modelo devolve um ranking top-5 e o bot apresenta a primeira sugestão
8. a primeira interação pós-recomendação mostra apenas `Gostei` ou `Não gostei`
9. depois do feedback, o bot mostra um menu limpo com `Outra sugestão`, `Sugestão do dia` e `Encerrar`
10. o feedback é salvo em `datasets/user_feedback.csv` com `user_id`, `timestamp`, década e popularidade da busca
11. as próximas recomendações do mesmo usuário passam a considerar histórico de curtidas, rejeições, diretores, gêneros e palavras-chave já vistos

Fluxo do `/recomendar`:

1. o bot consome a fila local gerada pela curadoria
2. entrega um filme por vez
3. remove o item consumido de `data/fila_curadoria.json`

## Drift com Evidently

O projeto mantém o baseline de treino em `datasets/reference_features.csv` e pode gerar o relatório HTML em `reports/relatorio_drift.html`.

Por padrão, o endpoint `POST /predict` **não** executa o Evidently durante a recomendação, para não aumentar a latência do bot. Se quiser habilitar essa geração no caminho da API, defina:

```env
CINEBOT_ENABLE_DRIFT_ON_PREDICT=true
```

Relatório esperado:

- `reports/relatorio_drift.html`
