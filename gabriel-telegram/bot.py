import json
import logging
import os
import sys
from pathlib import Path

import requests
import telebot
from dotenv import load_dotenv
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from botcity.maestro import AutomationTaskFinishStatus, BotMaestroSDK
from cinebot_ml.config import (
    DECADE_LABELS,
    DEFAULT_DATA_PATH,
    FEEDBACK_PATH,
    GENRE_LABELS,
    POPULARITY_LABELS,
)
from cinebot_ml.dataset import append_feedback

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("CINEBOT_ML_API_URL", "http://127.0.0.1:8000")
DATA_PATH = DEFAULT_DATA_PATH
LOCAL_QUEUE_PATH = Path(os.getenv("LOCAL_QUEUE_PATH", "data/fila_curadoria.json"))
USE_DATAPOOL = os.getenv("USE_DATAPOOL", "false").strip().lower() in {"1", "true", "yes", "sim"}
estado_usuarios = {}


def carregar_maestro() -> BotMaestroSDK | None:
    precisa_vault = USE_DATAPOOL or not os.getenv("TELEGRAM_BOT_TOKEN")
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


maestro = carregar_maestro()

GENEROS = GENRE_LABELS
DECADAS = DECADE_LABELS
POPULARIDADES = POPULARITY_LABELS


def formatar_mensagem_curadoria(filme: dict, origem: str) -> str:
    try:
        nota = float(filme.get("nota", 0))
    except (ValueError, TypeError):
        nota = 0.0

    linhas = [
        "SUGESTÃO DO DIA",
        "----------------",
        f"Título: {filme.get('titulo', 'Desconhecido')} ({filme.get('ano', 'N/A')})",
        f"Direção: {filme.get('diretor', 'Desconhecido')}",
        f"Duração: {filme.get('duracao', 0)} min",
        f"Nota: {nota:.1f} | Origem: {filme.get('nacionalidade', 'N/A')}",
        f"Gênero: {filme.get('genero', 'N/A')}",
        f"Onde assistir: {filme.get('streaming', 'N/A')}",
        "",
        f"Sinopse: {filme.get('sinopse', '')}",
        "",
        f"Link TMDB: {filme.get('url', '')}",
    ]
    return "\n".join(linhas)


def formatar_mensagem_ml(
    filme: dict,
    ranked_genres: list[str],
    decade_preference: str | None,
    popularity_preference: str | None,
) -> str:
    perfil_busca = [
        ", ".join(ranked_genres),
        decade_preference or "N/A",
        popularity_preference or "N/A",
    ]
    linhas = [
        "SUGESTÃO PARA VOCÊ",
        "------------------",
        f"Seu filtro desta busca: {' | '.join(perfil_busca)}",
        f"Título: {filme.get('titulo', 'Desconhecido')} ({filme.get('ano', 'N/A')})",
        f"Gêneros do filme: {', '.join(filme.get('generos', [])) or filme.get('perfil', 'N/A')}",
        f"Direção: {filme.get('diretor', 'Desconhecido')}",
        f"Nota TMDB: {filme.get('nota', 0)}",
        f"Onde assistir: {', '.join(filme.get('streaming', [])) or 'N/A'}",
        "",
        f"Sinopse: {filme.get('sinopse', '')}",
        "",
        f"Link TMDB: {filme.get('url', '')}",
    ]
    return "\n".join(linhas)


def gerar_teclado_generos(selecionados: list[str]) -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    for chave, label in GENEROS.items():
        if chave not in selecionados:
            markup.add(InlineKeyboardButton(label, callback_data=f"gen:{chave}"))
    return markup


def gerar_teclado_decadas() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    for chave, label in DECADAS.items():
        markup.add(InlineKeyboardButton(label, callback_data=f"dec:{chave}"))
    return markup


def gerar_teclado_popularidade() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    for chave, label in POPULARIDADES.items():
        markup.add(InlineKeyboardButton(label, callback_data=f"pop:{chave}"))
    return markup


def teclado_feedback_recomendacao_ml() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Gostei", callback_data="acao:like"))
    markup.add(InlineKeyboardButton("Não gostei", callback_data="acao:dislike"))
    return markup


def teclado_pos_feedback_ml() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Outra sugestão", callback_data="acao:proximo_ml"))
    markup.add(InlineKeyboardButton("Sugestão do dia", callback_data="acao:curadoria"))
    markup.add(InlineKeyboardButton("Encerrar", callback_data="acao:parar"))
    return markup


def teclado_sem_mais_sugestoes_ml() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Nova busca", callback_data="acao:reiniciar"))
    markup.add(InlineKeyboardButton("Sugestão do dia", callback_data="acao:curadoria"))
    markup.add(InlineKeyboardButton("Encerrar", callback_data="acao:parar"))
    return markup


def teclado_pos_curadoria() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Nova busca", callback_data="acao:reiniciar"))
    markup.add(InlineKeyboardButton("Outra sugestão do dia", callback_data="acao:curadoria"))
    markup.add(InlineKeyboardButton("Encerrar", callback_data="acao:parar"))
    return markup


def buscar_recomendacoes_ml(
    ranked_genres: list[str],
    decade_preference: str,
    popularity_preference: str,
    user_id: int | str | None = None,
) -> dict:
    payload = {
        "ranked_genres": ranked_genres,
        "decade_preference": decade_preference,
        "popularity_preference": popularity_preference,
        "data_path": str(DATA_PATH),
        "top_n": 5,
        "user_id": str(user_id) if user_id is not None else None,
    }
    response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = None
        if response.status_code == 400 and detail:
            raise ValueError(str(detail))
        response.raise_for_status()
    return response.json()


def iniciar_fluxo_ml(bot_tg: telebot.TeleBot, chat_id: int, user_id: int) -> None:
    estado_usuarios[user_id] = {
        "ranked_genres": [],
        "decade_preference": None,
        "popularity_preference": None,
        "recommendations": [],
        "current_movie": None,
        "drift_report": None,
        "awaiting_feedback": False,
    }
    bot_tg.send_message(
        chat_id,
        "Etapa 1 de 3: selecione o gênero que você mais gosta.",
        reply_markup=gerar_teclado_generos([]),
    )


def enviar_recomendacao_ml(bot_tg: telebot.TeleBot, chat_id: int, user_id: int) -> None:
    estado = estado_usuarios.get(user_id, {})
    fila = estado.get("recommendations", [])

    if not fila:
        bot_tg.send_message(
            chat_id,
            "Não há mais sugestões nessa busca. Toque em 'Nova busca' para escolher outros gêneros.",
            reply_markup=teclado_sem_mais_sugestoes_ml(),
        )
        return

    filme = fila.pop(0)
    estado["current_movie"] = filme
    estado["recommendations"] = fila
    estado["awaiting_feedback"] = True
    estado_usuarios[user_id] = estado

    mensagem = formatar_mensagem_ml(
        filme,
        estado.get("ranked_genres", []),
        estado.get("decade_preference"),
        estado.get("popularity_preference"),
    )
    markup = teclado_feedback_recomendacao_ml()

    if filme.get("poster"):
        bot_tg.send_photo(chat_id, photo=filme["poster"], caption=mensagem, reply_markup=markup)
    else:
        bot_tg.send_message(chat_id, mensagem, reply_markup=markup)


def registrar_feedback(user_id: int, feedback_value: str) -> bool:
    estado = estado_usuarios.get(user_id, {})
    filme = estado.get("current_movie")
    ranked_genres = estado.get("ranked_genres", [])
    decade_preference = estado.get("decade_preference")
    popularity_preference = estado.get("popularity_preference")

    if not filme or len(ranked_genres) != 3 or not estado.get("awaiting_feedback"):
        return False

    append_feedback(
        user_id=user_id,
        movie_id=int(filme["movie_id"]),
        ranked_genres=ranked_genres,
        decade_preference=decade_preference,
        popularity_preference=popularity_preference,
        feedback_value=feedback_value,
        feedback_path=FEEDBACK_PATH,
    )
    estado["awaiting_feedback"] = False
    estado_usuarios[user_id] = estado
    return True


def carregar_fila_local() -> list[dict]:
    if not LOCAL_QUEUE_PATH.exists():
        return []
    with open(LOCAL_QUEUE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def salvar_fila_local(fila: list[dict]) -> None:
    LOCAL_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_QUEUE_PATH, "w", encoding="utf-8") as handle:
        json.dump(fila, handle, ensure_ascii=False, indent=2)


def consumir_fila_local() -> dict | None:
    fila = carregar_fila_local()
    if not fila:
        return None
    filme = fila.pop(0)
    salvar_fila_local(fila)
    return filme


def comando_recomendar(bot_tg: telebot.TeleBot, message, execution) -> None:
    bot_tg.send_message(message.chat.id, "Buscando uma sugestão para você...")
    try:
        if USE_DATAPOOL:
            if maestro is None:
                raise RuntimeError("Maestro indisponível para uso com DataPool.")

            datapool = maestro.get_datapool("gabriel-filmes")
            task_id = execution.task_id if execution else None
            if not datapool.has_next():
                bot_tg.send_message(message.chat.id, "Não há mais sugestões disponíveis no momento.")
                return
            entry = datapool.next(task_id=task_id)
            filme = entry.values
        else:
            filme = consumir_fila_local()
            if not filme:
                bot_tg.send_message(message.chat.id, "Não há mais sugestões disponíveis no momento.")
                return
            entry = None

        mensagem = formatar_mensagem_curadoria(filme, "")

        if filme.get("poster"):
            bot_tg.send_photo(message.chat.id, photo=filme["poster"], caption=mensagem, reply_markup=teclado_pos_curadoria())
        else:
            bot_tg.send_message(message.chat.id, mensagem, reply_markup=teclado_pos_curadoria())

        if entry is not None:
            entry.report_done()
        logger.info("Fila consumida: %s", filme.get("titulo"))
    except Exception as exc:
        logger.error("Erro no /recomendar: %s", exc)
        bot_tg.send_message(message.chat.id, f"Erro ao consultar a fila: {exc}")


def main() -> None:
    global DATA_PATH
    logger.info("Bot Telegram iniciando")

    try:
        execution = obter_execucao(maestro)
        if execution and execution.parameters:
            data_path = execution.parameters.get("DATA_PATH")
            if data_path:
                DATA_PATH = Path(data_path)
        elif DEFAULT_DATA_PATH.exists():
            DATA_PATH = DEFAULT_DATA_PATH

        try:
            token = maestro.get_credential("gabriel-telegram", "token") if maestro else os.getenv("TELEGRAM_BOT_TOKEN")
        except Exception:
            token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("Credencial do Telegram não configurada no Vault nem no .env")

        bot_tg = telebot.TeleBot(token)

        @bot_tg.message_handler(commands=["start", "help"])
        def start(message):
            texto = (
                "CineBot de recomendações.\n\n"
                "COMANDOS:\n"
                "/recomendar - receber uma sugestão pronta\n"
                "/sugestao - escolher 3 gêneros e receber sugestões personalizadas"
            )
            bot_tg.reply_to(message, texto)

        @bot_tg.message_handler(commands=["recomendar"])
        def recomendar(message):
            comando_recomendar(bot_tg, message, execution)

        @bot_tg.message_handler(commands=["sugestao"])
        def sugestao(message):
            iniciar_fluxo_ml(bot_tg, message.chat.id, message.from_user.id)

        @bot_tg.callback_query_handler(func=lambda call: True)
        def callback_handler(call):
            user_id = call.from_user.id
            chat_id = call.message.chat.id
            message_id = call.message.message_id
            data = call.data

            if data.startswith("gen:"):
                genero = data.split(":", 1)[1]
                estado = estado_usuarios.setdefault(user_id, {"ranked_genres": []})
                ranked = estado.setdefault("ranked_genres", [])

                if genero in ranked:
                    bot_tg.answer_callback_query(call.id, "Gênero já selecionado.")
                    return

                ranked.append(genero)
                posicao = len(ranked)

                if posicao < 3:
                    bot_tg.edit_message_text(
                        f"Etapa 1 de 3: selecione o gênero número {posicao + 1} da sua preferência.",
                        chat_id=chat_id,
                        message_id=message_id,
                        reply_markup=gerar_teclado_generos(ranked),
                    )
                    return

                bot_tg.edit_message_text(
                    "Etapa 2 de 3: escolha a década do filme que você quer ver hoje.",
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=gerar_teclado_decadas(),
                )
                return

            if data.startswith("dec:"):
                decade_preference = data.split(":", 1)[1]
                estado = estado_usuarios.setdefault(user_id, {"ranked_genres": []})
                estado["decade_preference"] = DECADAS.get(decade_preference, decade_preference)
                estado_usuarios[user_id] = estado

                bot_tg.edit_message_text(
                    "Etapa 3 de 3: escolha entre um filme popular ou uma joia escondida.",
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=gerar_teclado_popularidade(),
                )
                return

            if data.startswith("pop:"):
                popularity_preference = data.split(":", 1)[1]
                estado = estado_usuarios.setdefault(user_id, {"ranked_genres": []})
                ranked = estado.get("ranked_genres", [])
                decade_preference = estado.get("decade_preference")
                estado["popularity_preference"] = POPULARIDADES.get(popularity_preference, popularity_preference)
                estado_usuarios[user_id] = estado

                bot_tg.edit_message_text(
                    "Buscando sugestões personalizadas para você...",
                    chat_id=chat_id,
                    message_id=message_id,
                )

                try:
                    resposta = buscar_recomendacoes_ml(
                        ranked,
                        decade_preference=decade_preference,
                        popularity_preference=estado["popularity_preference"],
                        user_id=user_id,
                    )
                except ValueError as exc:
                    logger.info("Busca sem resultado exato: %s", exc)
                    bot_tg.send_message(
                        chat_id,
                        f"{exc}\n\nTente mudar a década, a popularidade ou iniciar uma nova busca.",
                        reply_markup=teclado_sem_mais_sugestoes_ml(),
                    )
                    return
                except Exception as exc:
                    logger.error("Falha ao consultar /predict: %s", exc)
                    bot_tg.send_message(chat_id, f"Não consegui buscar sugestões agora: {exc}")
                    return

                estado["ranked_genres"] = resposta.get("ranked_genres", ranked)
                estado["decade_preference"] = resposta.get("decade_preference", estado.get("decade_preference"))
                estado["popularity_preference"] = resposta.get(
                    "popularity_preference",
                    estado.get("popularity_preference"),
                )
                estado["recommendations"] = resposta.get("recommendations", [])
                estado["drift_report"] = resposta.get("drift_report")
                estado["current_movie"] = None
                estado["awaiting_feedback"] = False
                estado_usuarios[user_id] = estado

                enviar_recomendacao_ml(bot_tg, chat_id, user_id)
                return

            if data.startswith("acao:"):
                acao = data.split(":", 1)[1]
                bot_tg.edit_message_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=None)

                if acao == "proximo_ml":
                    enviar_recomendacao_ml(bot_tg, chat_id, user_id)
                    return

                if acao == "like":
                    sucesso = registrar_feedback(user_id, "like")
                    resposta = (
                        "Que bom! Vou levar isso em conta nas próximas sugestões."
                        if sucesso
                        else "Não encontrei uma sugestão ativa para registrar."
                    )
                    bot_tg.send_message(chat_id, resposta, reply_markup=teclado_pos_feedback_ml())
                    return

                if acao == "dislike":
                    sucesso = registrar_feedback(user_id, "dislike")
                    resposta = (
                        "Entendi. Vou tentar algo melhor nas próximas sugestões."
                        if sucesso
                        else "Não encontrei uma sugestão ativa para registrar."
                    )
                    bot_tg.send_message(chat_id, resposta, reply_markup=teclado_pos_feedback_ml())
                    return

                if acao == "curadoria":
                    comando_recomendar(bot_tg, call.message, execution)
                    return

                if acao == "reiniciar":
                    iniciar_fluxo_ml(bot_tg, chat_id, user_id)
                    return

                if acao == "parar":
                    bot_tg.send_message(chat_id, "Sessão encerrada. Digite /start quando quiser voltar.")

        bot_tg.infinity_polling()
    except Exception as exc:
        logger.error("Erro fatal na inicialização: %s", exc)
        try:
            execution = obter_execucao(maestro)
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
