import json
import os
import time
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from botcity.web import WebBot, Browser, By
from botcity.maestro import BotMaestroSDK, AutomationTaskFinishStatus, DataPoolEntry
from botcity.web.browsers.chrome import default_options

load_dotenv()

# Constantes

TMDB_BASE_URL = "https://www.themoviedb.org"
TMDB_API_URL  = "https://api.themoviedb.org/3"

PERFIS = {
    "acao":     {"genero_id": 28,    "genero_nome": "Ação"},
    "comedia":  {"genero_id": 35,    "genero_nome": "Comédia"},
    "drama":    {"genero_id": 18,    "genero_nome": "Drama"},
    "terror":   {"genero_id": 27,    "genero_nome": "Terror"},
    "scifi":    {"genero_id": 878,   "genero_nome": "Ficção Científica"},
    "romance":  {"genero_id": 10749, "genero_nome": "Romance"},
    "suspense": {"genero_id": 53,    "genero_nome": "Suspense"},
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

# API REST 

def buscar_filmes_por_perfil(perfil: dict, api_key: str) -> list:
    params = {
        "api_key": api_key,
        "language": "pt-BR",
        "with_genres": perfil["genero_id"],
        "vote_average.gte": 7.2,
        "vote_count.gte": 800,
        "vote_count.lte": 7000,
        "sort_by": "vote_average.desc",
        "without_genres": "16,10751,10770",
        "with_original_language": "en|pt|fr|de|it|es|ja|ko",
        "with_runtime.gte": 90,
        "page": 1,
    }
    try:
        r = requests.get(f"{TMDB_API_URL}/discover/movie", params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except requests.RequestException as e:
        print(f"[scraper] Erro na API: {e}")
        return []


def buscar_detalhes_api(filme_id: int, api_key: str) -> dict:
    try:
        r = requests.get(
            f"{TMDB_API_URL}/movie/{filme_id}",
            params={"api_key": api_key, "language": "pt-BR", "append_to_response": "credits"},
            timeout=10,
        )
        return r.json()
    except Exception:
        return {}


def buscar_streaming(filme_id: int, api_key: str) -> list:
    try:
        r = requests.get(
            f"{TMDB_API_URL}/movie/{filme_id}/watch/providers",
            params={"api_key": api_key},
            timeout=10,
        )
        br = r.json().get("results", {}).get("BR", {})
        return [p["provider_name"] for p in br.get("flatrate", [])[:4]]
    except Exception:
        return []


# BotCity Web: extrai título e sinopse 

def extrair_com_botcity(bot: WebBot, url: str, filme_api: dict) -> tuple[str, str]:
    try:
        bot.browse(url)
        time.sleep(3)

        titulo = filme_api.get("title", "")
        try:
            el = bot.find_element("h2.title a, div.title h2 a", By.CSS_SELECTOR)
            if el:
                titulo = el.text.strip() or titulo
        except Exception:
            pass

        sinopse = filme_api.get("overview", "")
        try:
            el = bot.find_element("div.overview p", By.CSS_SELECTOR)
            if el:
                sinopse = el.text.strip() or sinopse
        except Exception:
            pass

        if len(sinopse) > 700:
            sinopse = sinopse[:697] + "..."

        return titulo, sinopse

    except Exception as e:
        print(f"[scraper] Erro BotCity em {url}: {e}")
        return filme_api.get("title", ""), filme_api.get("overview", "")


# Salvar JSON 

def salvar(dados: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    print(f"[scraper] Dados salvos em {path}")


# Entry point 

def main():
    load_dotenv()
    maestro = BotMaestroSDK.from_sys_args()

    if not maestro.server:
        maestro.login(
            server=os.getenv("MAESTRO_SERVER"),
            login=os.getenv("MAESTRO_LOGIN"),
            key=os.getenv("MAESTRO_KEY")
        )

    execution = maestro.get_execution()
    parameters = execution.parameters

    data_path = parameters.get("DATA_PATH")
    if not data_path:
        raise ValueError("Parametro DATA_PATH nao configurado no Maestro")
    DATA_PATH = Path(data_path)

    print(f"[scraper] Iniciando coleta — {datetime.now()}")

    try:
        try:
            tmdb_key = maestro.get_credential("gabriel-tmdb", "api_key")
        except Exception:
            tmdb_key = os.getenv("TMDB_API_KEY")

        if not tmdb_key:
            raise ValueError("TMDB_API_KEY nao configurada")

        todos_filmes = {}

        bot = WebBot()
        bot.headless = False
        bot.browser = Browser.CHROME
        bot.driver_path = "/usr/bin/chromedriver"
        bot._options = None
        opts = default_options(headless=False, download_folder_path=None, user_data_dir=None)
        opts.binary_location = "/usr/bin/chromium-browser"
        bot.options = opts
        bot.start_browser()

        for chave, perfil in PERFIS.items():
            print(f"\n[scraper] Genero: {perfil['genero_nome']}")
            filmes_api = buscar_filmes_por_perfil(perfil, tmdb_key)

            if not filmes_api:
                todos_filmes[chave] = []
                continue

            filmes_detalhados = []
            ids_vistos = set()

            for filme_api in filmes_api[:1]:
                filme_id  = filme_api.get("id")
                url_filme = f"{TMDB_BASE_URL}/movie/{filme_id}"

                if filme_id in ids_vistos:
                    continue
                ids_vistos.add(filme_id)

                titulo, sinopse = extrair_com_botcity(bot, url_filme, filme_api)

                detalhes   = buscar_detalhes_api(filme_id, tmdb_key)
                paises     = detalhes.get("production_countries", [])
                nac        = ", ".join([PAISES_PT.get(p["name"], p["name"]) for p in paises[:2]]) or "N/A"
                equipe     = detalhes.get("credits", {}).get("crew", [])
                diretor    = next((c["name"] for c in equipe if c["job"] == "Director"), "Desconhecido")
                duracao    = detalhes.get("runtime", 0)
                streaming  = buscar_streaming(filme_id, tmdb_key)
                data_str   = filme_api.get("release_date", "")
                ano        = data_str[:4] if data_str else "N/A"
                poster     = ""
                if filme_api.get("poster_path"):
                    poster = f"https://image.tmdb.org/t/p/w500{filme_api['poster_path']}"

                filme = {
                    "id":            filme_id,
                    "titulo":        titulo,
                    "sinopse":       sinopse,
                    "ano":           ano,
                    "duracao":       duracao,
                    "diretor":       diretor,
                    "nacionalidade": nac,
                    "streaming":     streaming,
                    "nota":          round(filme_api.get("vote_average", 0.0), 1),
                    "votos":         filme_api.get("vote_count", 0),
                    "url":           url_filme,
                    "poster":        poster,
                    "perfil":        chave,
                    "genero":        perfil["genero_nome"],
                }

                filmes_detalhados.append(filme)
                time.sleep(2)

            todos_filmes[chave] = filmes_detalhados
            print(f"[scraper] {len(filmes_detalhados)} filmes para '{chave}'")
            time.sleep(3)

        bot.stop_browser()

        payload = {
            "atualizado_em": datetime.now().isoformat(),
            "perfis": todos_filmes,
        }
        salvar(payload, DATA_PATH)

        maestro.post_artifact(
            task_id=execution.task_id,
            artifact_name="filmes.json",
            filepath=str(DATA_PATH),
        )
        maestro.finish_task(
            task_id=execution.task_id,
            status=AutomationTaskFinishStatus.SUCCESS,
            message=f"Scraper concluido. {sum(len(v) for v in todos_filmes.values())} filmes coletados.",
        )

        print(f"[scraper] Coleta finalizada — {datetime.now()}")

    except Exception as e:
        print(f"[scraper] Erro fatal: {e}")
        try:
            maestro.finish_task(
                task_id=execution.task_id,
                status=AutomationTaskFinishStatus.FAILED,
                message=str(e),
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()