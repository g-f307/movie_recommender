# Movie Recommender Bots

Sistema de curadoria cinematográfica automatizada com três bots integrados ao BotCity Maestro. O projeto coleta filmes do TMDB, organiza uma fila de recomendação e disponibiliza sugestões ao usuário por meio de um bot no Telegram.

## Visão geral

A solução foi organizada em três componentes principais:

- `gabriel-scrapper`: coleta filmes e metadados no TMDB e gera uma base local em JSON.
- `gabriel-curadoria`: lê a base gerada pelo scraper, evita duplicidades e publica os filmes no DataPool do Maestro.
- `gabriel-telegram`: entrega recomendações ao usuário, consumindo a fila do Maestro ou filtrando a base local.

Fluxo resumido:

1. O scraper consulta a API do TMDB e consolida os resultados em `data/filmes.json`.
2. A curadoria identifica os filmes ainda não enviados e publica novos registros no DataPool.
3. O bot do Telegram recomenda filmes diretamente da fila ou por filtros interativos.

## Estrutura do projeto

```text
.
├── data/
│   ├── curadoria.json
│   └── filmes.json
├── gabriel-curadoria/
│   ├── bot.py
│   └── requirements.txt
├── gabriel-scrapper/
│   ├── bot.py
│   ├── requirements.txt
│   └── resources/
├── gabriel-telegram/
│   ├── bot.py
│   └── requirements.txt
└── README.md
```

## Pré-requisitos

Antes de executar o projeto, tenha disponível:

- Python 3.10 ou superior
- `pip`
- Git
- conta ativa no BotCity Maestro
- token de bot do Telegram
- chave de API do TMDB
- Chromium ou Google Chrome instalado
- ChromeDriver compatível com a versão do navegador

Também é recomendável utilizar um ambiente virtual dedicado ao projeto.

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/g-f307/movie_recommender.git
cd movie_recommender
```

### 2. Criar e ativar um ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r gabriel-scrapper/requirements.txt
pip install -r gabriel-curadoria/requirements.txt
pip install -r gabriel-telegram/requirements.txt
```

## Configuração de credenciais

O projeto utiliza configuração local via `.env` e credenciais armazenadas no BotCity Maestro.

### Arquivo `.env`

Crie um arquivo `.env` na raiz do projeto:

```env
MAESTRO_SERVER=https://SEU_SERVIDOR_MAESTRO
MAESTRO_LOGIN=SEU_LOGIN
MAESTRO_KEY=SUA_CHAVE
TMDB_API_KEY=SUA_TMDB_API_KEY
```

Observações:

- `TMDB_API_KEY` funciona como fallback local para o scraper.
- as credenciais do Maestro são necessárias para execução fora do ambiente orquestrado.

### Credentials Vault no Maestro

Cadastre as credenciais esperadas pelo código atual:

#### TMDB

- credencial: `gabriel-tmdb`
- campo: `api_key`

#### Telegram

- credencial: `gabriel-telegram`
- campo: `token`

Se você preferir usar outros nomes, será necessário ajustar o código para refletir os novos identificadores cadastrados no Maestro.

## Configuração no BotCity Maestro

Para executar a solução na plataforma, configure os seguintes recursos.

### 1. Registro dos bots

Cadastre separadamente os bots:

- `gabriel-scrapper`
- `gabriel-curadoria`
- `gabriel-telegram`

Cada pacote deve conter, no mínimo:

- `bot.py`
- `requirements.txt`
- `resources/`, quando aplicável

### 2. DataPool

O código atual utiliza um DataPool com identificador:

```text
gabriel-filmes
```

Esse DataPool é usado pela curadoria para inserir novos filmes e pelo bot do Telegram para consumir a fila de recomendações.

Se o ambiente já possuir um identificador diferente ou se esse nome não puder ser utilizado, altere o valor correspondente no código antes do deploy.

### 3. Parâmetro de execução

Os bots que leem ou gravam o catálogo dependem do parâmetro:

```text
DATA_PATH
```

Exemplo:

```text
data/filmes.json
```

Ajuste o caminho conforme a estratégia de armazenamento adotada na execução.

### 4. Artifacts

Os bots publicam os seguintes artefatos:

- scraper: `filmes.json`
- curadoria: `historico_inseridos.json`

## Execução local

### Scraper

Responsável por coletar filmes por perfil e gerar o catálogo local.

```bash
python gabriel-scrapper/bot.py
```

Saída esperada:

- criação ou atualização de `data/filmes.json`
- coleta de título, sinopse, ano, diretor, streaming, nota e poster

### Curadoria

Responsável por ler o catálogo do scraper e publicar os novos filmes no DataPool.

```bash
python gabriel-curadoria/bot.py
```

Saída esperada:

- leitura de `data/filmes.json`
- inserção de novos registros no DataPool
- atualização de `historico_inseridos.json`

### Telegram

Responsável por recomendar filmes via comandos e filtros interativos.

```bash
python gabriel-telegram/bot.py
```

Comandos disponíveis:

- `/start`
- `/help`
- `/recomendar`
- `/sugestao`

## Execução no BotCity Maestro

Ordem recomendada:

1. Registrar os três bots no Maestro.
2. Configurar as credenciais no Vault.
3. Criar ou ajustar o DataPool utilizado pelo projeto.
4. Definir o parâmetro `DATA_PATH`.
5. Executar o bot `gabriel-scrapper`.
6. Executar o bot `gabriel-curadoria`.
7. Executar o bot `gabriel-telegram`.

## Funcionalidades por bot

### `gabriel-scrapper`

- busca filmes por perfis de gênero
- utiliza a API do TMDB para descoberta e detalhamento
- consulta provedores de streaming
- extrai título e sinopse via BotCity Web
- gera a base consolidada `data/filmes.json`

### `gabriel-curadoria`

- carrega a base local criada pelo scraper
- verifica duplicidade por histórico de inserção
- cria entradas no DataPool do Maestro
- mantém o histórico de filmes já enviados

### `gabriel-telegram`

- conecta ao Telegram com token armazenado no Maestro
- recomenda filmes pendentes da fila
- oferece sugestão por gênero, época e estilo
- utiliza o arquivo local como fallback de recomendação

## Tecnologias utilizadas

- Python
- BotCity Framework Web
- BotCity Maestro SDK
- Selenium
- Requests
- Python Dotenv
- PyTelegramBotAPI
- TMDB API
- JSON para persistência local

## Observações importantes

- Execute primeiro o scraper para garantir a existência de `data/filmes.json`.
- Execute a curadoria antes do Telegram se quiser demonstrar o consumo do DataPool.
- Verifique se o ChromeDriver é compatível com a versão do navegador instalado.
- Credenciais e identificadores configurados no Maestro devem estar alinhados com os valores utilizados pelo código.
