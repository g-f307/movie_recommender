# Movie Recommender Bots

> **Propriedade Intelectual:** Este projeto e todo o seu código-fonte, modelos e artefatos são de propriedade da LG Electronics do Brasil Ltda., desenvolvido no âmbito do projeto AX Academy --- Digital Transformation (Convênio N.º 005/2025 --- INOVA / IFAM). Consulte o arquivo LICENSE para mais detalhes.

Upgrade do `CineBot, curadoria de filmes` para o `Desafio 02 - Upgrade do Bot Pessoal com ML`. O projeto preserva os três bots originais do Módulo 1 e adiciona uma camada de Machine Learning para recomendar filmes a partir de `3 gêneros ranqueados pelo usuário`, mais filtros de `década` e `popularidade`, como indicado na Seção 8 do enunciado para `Gabriel de Sá`.

Nesta versão, o enquadramento metodológico é: `modelo supervisionado de aderência ao perfil declarado` para cold start, complementado por `personalização incremental com feedback real por user_id`.

## Arquitetura

O fluxo agora ficou dividido em seis partes:

1. `gabriel-scrapper/`
   Coleta filmes do TMDB com BotCity Web, prioriza sinopses em `pt-BR`, limita excesso de franquias via coleção do TMDB, extrai diretor, gêneros, streaming e gera `data/filmes.json`.
2. `gabriel-curadoria/`
   Lê a base coletada e monta uma fila local em `data/fila_curadoria.json`. O envio ao DataPool do Maestro ficou opcional.
3. `cinebot_ml/dataset.py`
   Gera o dataset supervisionado `datasets/movie_preferences.csv` a partir do catálogo coletado e dos perfis de preferência, marcando quando o rótulo vem do proxy heurístico e quando foi sobrescrito por feedback real.
4. `cinebot_ml/train.py`
   Compara `3 algoritmos distintos` com `Pipeline + ColumnTransformer`, aplica validação cruzada por grupos no treino, registra experimentos no MLflow e promove o vencedor para `@production`.
5. `main.py`
   Sobe uma API FastAPI com `GET /saude` e `POST /predict`.
6. `gabriel-telegram/`
   Recebe `3 gêneros ranqueados`, `década` e `popularidade`, chama o endpoint de ML, obtém um ranking top-5, apresenta uma sugestão por vez, salva feedback por `user_id` e personaliza as próximas recomendações daquele usuário.

## Estrutura

```text
.
├── .dvc/
├── cinebot_ml/
├── data/
├── datasets/
├── gabriel-curadoria/
├── gabriel-scrapper/
├── gabriel-telegram/
├── dvc.yaml
├── LICENSE
├── main.py
├── README.md
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

- Python `3.10+`
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

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r gabriel-scrapper/requirements.txt
pip install -r gabriel-curadoria/requirements.txt
pip install -r gabriel-telegram/requirements.txt
pip install -r requirements-ml.txt
```

## Configuração

Crie um `.env` na raiz:

```env
MAESTRO_SERVER=https://SEU_SERVIDOR_MAESTRO
MAESTRO_LOGIN=SEU_LOGIN
MAESTRO_KEY=SUA_CHAVE
TMDB_API_KEY=SUA_TMDB_API_KEY
TELEGRAM_BOT_TOKEN=SEU_TOKEN_LOCAL_OPCIONAL
CINEBOT_ML_API_URL=http://127.0.0.1:8000
DATA_PATH=data/filmes.json
MOVIES_PER_PROFILE=0
MAX_MOVIES_PER_COLLECTION=1
MIN_VOTE_AVERAGE=7.2
MIN_VOTE_COUNT=300
DISCOVER_SORT_BY=popularity.desc
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

Ao processar uma recomendação via API, o projeto tenta comparar os dados atuais com o baseline de treino salvo em `datasets/reference_features.csv`.

Relatório esperado:

- `reports/relatorio_drift.html`
