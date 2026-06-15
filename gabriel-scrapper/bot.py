import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from botcity.maestro import BotMaestroSDK, AutomationTaskFinishStatus
from botcity.web import Browser, By, WebBot
from botcity.web.browsers.chrome import default_options

load_dotenv()

TMDB_BASE_URL = "https://www.themoviedb.org"
TMDB_API_URL = "https://api.themoviedb.org/3"

PERFIS = {
    "acao": {"genero_id": 28, "genero_nome": "Ação"},
    "comedia": {"genero_id": 35, "genero_nome": "Comédia"},
    "drama": {"genero_id": 18, "genero_nome": "Drama"},
    "terror": {"genero_id": 27, "genero_nome": "Terror"},
    "scifi": {"genero_id": 878, "genero_nome": "Ficção Científica"},
    "romance": {"genero_id": 10749, "genero_nome": "Romance"},
    "suspense": {"genero_id": 53, "genero_nome": "Suspense"},
}

PAISES_PT = {
    "United States of America": "Estados Unidos",
    "United Kingdom": "Reino Unido",
    "France": "França",
    "Germany": "Alemanha",
    "Japan": "Japão",
    "Brazil": "Brasil",
    "Italy": "Itália",
    "South Korea": "Coreia do Sul",
    "Mexico": "México",
    "Argentina": "Argentina",
    "Canada": "Canadá",
    "Sweden": "Suécia",
    "Australia": "Austrália",
}

GENRE_ID_TO_KEY = {perfil["genero_id"]: chave for chave, perfil in PERFIS.items()}


def parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "sim"}


def buscar_filmes_por_perfil(
    perfil: dict,
    api_key: str,
    page: int = 1,
    min_vote_average: float = 7.2,
    min_vote_count: int = 300,
    sort_by: str = "popularity.desc",
) -> list:
    params = {
        "api_key": api_key,
        "language": "pt-BR",
        "with_genres": perfil["genero_id"],
        "vote_average.gte": min_vote_average,
        "vote_count.gte": min_vote_count,
        "sort_by": sort_by,
        "without_genres": "16,10751,10770",
        "with_original_language": "en|pt|fr|de|it|es|ja|ko",
        "with_runtime.gte": 80,
        "primary_release_date.lte": datetime.now().date().isoformat(),
        "page": page,
    }
    try:
        response = requests.get(f"{TMDB_API_URL}/discover/movie", params=params, timeout=20)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.RequestException as exc:
        print(f"[scraper] Erro na API para {perfil['genero_nome']}: {exc}")
        return []


def buscar_detalhes_api(filme_id: int, api_key: str) -> dict:
    try:
        response = requests.get(
            f"{TMDB_API_URL}/movie/{filme_id}",
            params={
                "api_key": api_key,
                "language": "pt-BR",
                "append_to_response": "credits,keywords",
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {}


def buscar_streaming(filme_id: int, api_key: str) -> list:
    try:
        response = requests.get(
            f"{TMDB_API_URL}/movie/{filme_id}/watch/providers",
            params={"api_key": api_key},
            timeout=20,
        )
        response.raise_for_status()
        providers = response.json().get("results", {}).get("BR", {})
        return [provider["provider_name"] for provider in providers.get("flatrate", [])[:4]]
    except requests.RequestException:
        return []


def inferir_genero_principal(chave_atual: str, detalhes: dict) -> str | None:
    for genero in detalhes.get("genres", []) or []:
        genero_id = genero.get("id")
        if genero_id in GENRE_ID_TO_KEY:
            return GENRE_ID_TO_KEY[genero_id]

    genero_ids = {genero.get("id") for genero in detalhes.get("genres", []) or []}
    if PERFIS[chave_atual]["genero_id"] in genero_ids:
        return chave_atual
    return None


def extrair_com_botcity(bot: WebBot, url: str, filme_api: dict) -> tuple[str, str]:
    try:
        bot.browse(url)
        time.sleep(2)

        titulo = filme_api.get("title", "")
        sinopse = filme_api.get("overview", "")

        try:
            title_element = bot.find_element("h2.title a, div.title h2 a", By.CSS_SELECTOR)
            if title_element and title_element.text.strip():
                titulo = title_element.text.strip()
        except Exception:
            pass

        try:
            overview_element = bot.find_element("div.overview p", By.CSS_SELECTOR)
            if overview_element and overview_element.text.strip():
                sinopse = overview_element.text.strip()
        except Exception:
            pass

        if len(sinopse) > 700:
            sinopse = sinopse[:697] + "..."

        return titulo, sinopse
    except Exception as exc:
        print(f"[scraper] Erro BotCity em {url}: {exc}")
        return filme_api.get("title", ""), filme_api.get("overview", "")


def salvar(dados: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dados, handle, ensure_ascii=False, indent=2)
    print(f"[scraper] Dados salvos em {path}")


def criar_bot(headless: bool) -> WebBot:
    bot = WebBot()
    bot.headless = headless
    bot.browser = Browser.CHROME
    bot.driver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
    bot._options = None

    options = default_options(headless=headless, download_folder_path=None, user_data_dir=None)
    options.binary_location = os.getenv("CHROMIUM_BINARY", "/usr/bin/chromium-browser")
    bot.options = options
    bot.start_browser()
    return bot


def carregar_maestro() -> BotMaestroSDK | None:
    precisa_vault = not os.getenv("TMDB_API_KEY")
    if not precisa_vault:
        return None

    try:
        maestro = BotMaestroSDK.from_sys_args()
        if maestro.server:
            return maestro
    except Exception:
        maestro = BotMaestroSDK()

    if os.getenv("MAESTRO_SERVER"):
        maestro.login(
            server=os.getenv("MAESTRO_SERVER"),
            login=os.getenv("MAESTRO_LOGIN"),
            key=os.getenv("MAESTRO_KEY"),
        )
        return maestro
    return None


def obter_execucao(maestro: BotMaestroSDK | None):
    if maestro is None:
        return None
    try:
        return maestro.get_execution()
    except Exception:
        return None


def main() -> None:
    load_dotenv()
    maestro = carregar_maestro()
    execution = obter_execucao(maestro)
    parameters = execution.parameters if execution else {}

    data_path = parameters.get("DATA_PATH") or os.getenv("DATA_PATH", "data/filmes.json")
    max_movies = int(parameters.get("MOVIES_PER_PROFILE") or os.getenv("MOVIES_PER_PROFILE", "0"))
    max_pages = int(parameters.get("MAX_PAGES") or os.getenv("MAX_PAGES", "8"))
    max_movies_per_collection = int(
        parameters.get("MAX_MOVIES_PER_COLLECTION") or os.getenv("MAX_MOVIES_PER_COLLECTION", "1")
    )
    min_vote_average = float(parameters.get("MIN_VOTE_AVERAGE") or os.getenv("MIN_VOTE_AVERAGE", "7.2"))
    min_vote_count = int(parameters.get("MIN_VOTE_COUNT") or os.getenv("MIN_VOTE_COUNT", "300"))
    discover_sort_by = parameters.get("DISCOVER_SORT_BY") or os.getenv("DISCOVER_SORT_BY", "popularity.desc")
    headless = parse_bool(parameters.get("HEADLESS") or os.getenv("HEADLESS"), default=True)
    data_output = Path(data_path)

    print(f"[scraper] Iniciando coleta em {datetime.now().isoformat()} | destino={data_output}")
    print(
        "[scraper] Configuração: "
        f"movies_per_profile={'sem cota rígida' if max_movies <= 0 else max_movies}, "
        f"max_pages={max_pages}, "
        f"max_movies_per_collection={max_movies_per_collection}, "
        f"min_vote_average={min_vote_average}, "
        f"min_vote_count={min_vote_count}, "
        f"sort_by={discover_sort_by}"
    )

    try:
        try:
            tmdb_key = maestro.get_credential("gabriel-tmdb", "api_key") if maestro else os.getenv("TMDB_API_KEY")
        except Exception:
            tmdb_key = os.getenv("TMDB_API_KEY")

        if not tmdb_key:
            raise ValueError("TMDB_API_KEY não configurada no Vault nem no .env")

        bot = criar_bot(headless=headless)
        todos_filmes = {}
        ids_globais = set()
        colecoes_globais = {}

        try:
            for chave, perfil in PERFIS.items():
                print(f"[scraper] Coletando gênero {perfil['genero_nome']}")
                page = 1
                filmes_detalhados = []
                ids_vistos = set()

                while page <= max_pages and (max_movies <= 0 or len(filmes_detalhados) < max_movies):
                    filmes_api = buscar_filmes_por_perfil(
                        perfil,
                        tmdb_key,
                        page=page,
                        min_vote_average=min_vote_average,
                        min_vote_count=min_vote_count,
                        sort_by=discover_sort_by,
                    )
                    page += 1

                    if not filmes_api:
                        continue

                    for filme_api in filmes_api:
                        filme_id = filme_api.get("id")
                        if not filme_id or filme_id in ids_vistos or filme_id in ids_globais:
                            continue

                        detalhes = buscar_detalhes_api(filme_id, tmdb_key)
                        genero_principal = inferir_genero_principal(chave, detalhes)
                        if genero_principal != chave:
                            continue

                        colecao = detalhes.get("belongs_to_collection") or {}
                        colecao_id = colecao.get("id")
                        if colecao_id and colecoes_globais.get(colecao_id, 0) >= max_movies_per_collection:
                            continue

                        ids_vistos.add(filme_id)
                        ids_globais.add(filme_id)
                        if colecao_id:
                            colecoes_globais[colecao_id] = colecoes_globais.get(colecao_id, 0) + 1

                        url_filme = f"{TMDB_BASE_URL}/movie/{filme_id}?language=pt-BR"

                        titulo_raspado, sinopse_raspada = extrair_com_botcity(bot, url_filme, filme_api)
                        titulo = detalhes.get("title") or filme_api.get("title") or titulo_raspado
                        sinopse = detalhes.get("overview") or filme_api.get("overview") or sinopse_raspada
                        if len(sinopse) > 700:
                            sinopse = sinopse[:697] + "..."
                        paises = detalhes.get("production_countries", [])
                        nacionalidade = ", ".join(PAISES_PT.get(item["name"], item["name"]) for item in paises[:2]) or "N/A"
                        equipe = detalhes.get("credits", {}).get("crew", [])
                        diretor = next((crew["name"] for crew in equipe if crew["job"] == "Director"), "Desconhecido")
                        duracao = detalhes.get("runtime", 0) or 0
                        streaming = buscar_streaming(filme_id, tmdb_key)
                        ano = (filme_api.get("release_date") or "")[:4] or "N/A"
                        generos_secundarios = [genre["name"] for genre in detalhes.get("genres", []) if genre.get("name")]
                        keywords = [
                            item["name"]
                            for item in detalhes.get("keywords", {}).get("keywords", [])[:6]
                            if item.get("name")
                        ]
                        poster = ""
                        if filme_api.get("poster_path"):
                            poster = f"https://image.tmdb.org/t/p/w500{filme_api['poster_path']}"

                        filme = {
                            "id": filme_id,
                            "titulo": titulo,
                            "sinopse": sinopse,
                            "ano": ano,
                            "duracao": duracao,
                            "diretor": diretor,
                            "nacionalidade": nacionalidade,
                            "streaming": streaming,
                            "nota": round(filme_api.get("vote_average", 0.0), 1),
                            "votos": filme_api.get("vote_count", 0),
                            "url": url_filme,
                            "poster": poster,
                            "perfil": chave,
                            "genero": perfil["genero_nome"],
                            "generos_secundarios": generos_secundarios,
                            "palavras_chave": keywords,
                            "genero_principal_id": PERFIS[chave]["genero_id"],
                            "collection_id": colecao_id,
                            "collection_name": colecao.get("name", ""),
                        }
                        filmes_detalhados.append(filme)

                        if max_movies > 0 and len(filmes_detalhados) >= max_movies:
                            break

                        time.sleep(1.2)

                todos_filmes[chave] = filmes_detalhados
                print(f"[scraper] {len(filmes_detalhados)} filmes preparados para {chave}")
                time.sleep(1)
        finally:
            bot.stop_browser()

        payload = {
            "atualizado_em": datetime.now().isoformat(),
            "configuracao": {
                "movies_per_profile": max_movies,
                "max_movies_per_collection": max_movies_per_collection,
                "min_vote_average": min_vote_average,
                "min_vote_count": min_vote_count,
                "discover_sort_by": discover_sort_by,
                "headless": headless,
            },
            "perfis": todos_filmes,
        }
        salvar(payload, data_output)

        if execution and maestro:
            maestro.post_artifact(
                task_id=execution.task_id,
                artifact_name="filmes.json",
                filepath=str(data_output),
            )
            maestro.finish_task(
                task_id=execution.task_id,
                status=AutomationTaskFinishStatus.SUCCESS,
                message=f"Scraper concluído. {sum(len(items) for items in todos_filmes.values())} filmes coletados.",
            )

        print(f"[scraper] Coleta finalizada em {datetime.now().isoformat()}")
    except Exception as exc:
        print(f"[scraper] Erro fatal: {exc}")
        try:
            if execution and maestro:
                maestro.finish_task(
                    task_id=execution.task_id,
                    status=AutomationTaskFinishStatus.FAILED,
                    message=str(exc),
                )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
