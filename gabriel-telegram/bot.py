import json
import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from botcity.maestro import BotMaestroSDK, AutomationTaskFinishStatus

load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATA_PATH = None

# MAESTRO SDK (GLOBAL)
try:
    maestro = BotMaestroSDK.from_sys_args()
except Exception:
    maestro = BotMaestroSDK(
        server=os.getenv("MAESTRO_SERVER"),
        login=os.getenv("MAESTRO_LOGIN"),
        key=os.getenv("MAESTRO_KEY")
    )
    maestro.login(
        server=os.getenv("MAESTRO_SERVER"),
        login=os.getenv("MAESTRO_LOGIN"),
        key=os.getenv("MAESTRO_KEY")
    )

# DICIONÁRIOS DO FUNIL 
GENEROS = {
    "acao": "Ação", "comedia": "Comédia", "drama": "Drama",
    "terror": "Terror", "scifi": "Ficção Científica", 
    "romance": "Romance", "suspense": "Suspense"
}
EPOCAS = {
    "classico": "Clássico (Antes de 2000)", "anos2000": "Anos 2000 (2000-2019)",
    "lancamento": "Lançamento (2020+)", "qualquer": "Sem preferência"
}
ESTILOS = {
    "lado_b": "Lado B (Joia Escondida)", "aclamado": "Aclamado (Sucesso de Crítica)"
}

estado_usuarios = {}

# HELPERS
def formatar_mensagem_tecnica(filme: dict, origem: str) -> str:
    try:
        nota = float(filme.get('nota', 0))
    except (ValueError, TypeError):
        nota = 0.0

    linhas = [
        f"RESULTADO DA CURADORIA ({origem})",
        "-----------------------------------",
        f"Título: {filme.get('titulo', 'Desconhecido')} ({filme.get('ano', 'N/A')})",
        f"Direção: {filme.get('diretor', 'Desconhecido')}",
        f"Duração: {filme.get('duracao', 0)} min",
        f"Nota: {nota:.1f} | Origem: {filme.get('nacionalidade', 'N/A')}",
        f"Gênero: {filme.get('genero', 'N/A')}",
        f"Onde assistir: {filme.get('streaming', 'N/A')}",
        "",
        f"Sinopse: {filme.get('sinopse', '')}",
        "",
        f"Link TMDB: {filme.get('url', '')}"
    ]
    return "\n".join(linhas)

def buscar_filme_local(genero, epoca, estilo) -> dict:
    global DATA_PATH
    if not DATA_PATH or not DATA_PATH.exists():
        logger.error(f"Arquivo de dados nao encontrado no caminho: {DATA_PATH}")
        return None
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dados = json.load(f)
        
    filmes_base = dados.get("perfis", {}).get(genero, [])
    if not filmes_base: return None

    filtrados = []
    if epoca and epoca != "qualquer":
        for f in filmes_base:
            ano = int(f.get("ano", 0)) if str(f.get("ano", "")).isdigit() else 0
            if epoca == "classico" and 0 < ano < 2000: filtrados.append(f)
            elif epoca == "anos2000" and 2000 <= ano <= 2019: filtrados.append(f)
            elif epoca == "lancamento" and ano >= 2020: filtrados.append(f)
    if not filtrados: filtrados = filmes_base

    if estilo == "aclamado":
        filtrados.sort(key=lambda x: x.get("votos", 0), reverse=True)
    elif estilo == "lado_b":
        filtrados.sort(key=lambda x: x.get("votos", 0))
    else:
        random.shuffle(filtrados)

    return random.choice(filtrados[:5]) if filtrados else None

def gerar_teclado(dicionario: dict, prefixo: str) -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    for chave, label in dicionario.items():
        markup.add(InlineKeyboardButton(label, callback_data=f"{prefixo}:{chave}"))
    return markup

def teclado_pos_sugestao() -> InlineKeyboardMarkup:
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Próximo da Fila (Maestro)", callback_data="acao:recomendar"))
    markup.add(InlineKeyboardButton("Novo Filtro (Q&A)", callback_data="acao:sugestao"))
    markup.add(InlineKeyboardButton("Encerrar Sessão", callback_data="acao:parar"))
    return markup

# ENTRY POINT
def main():
    global DATA_PATH
    logger.info("Bot Telegram iniciando")
    
    try:
        # 1. Captura de parâmetros
        execution = maestro.get_execution()
        
        if execution and execution.parameters:
            data_path = execution.parameters.get("DATA_PATH")
            if not data_path:
                raise ValueError("Parametro DATA_PATH nao configurado no Maestro")
                
            DATA_PATH = Path(data_path)
            logger.info(f"DATA_PATH carregado: {DATA_PATH}")
        else:
            logger.warning("Nao ha execucao ou parametros no Maestro. DATA_PATH indisponivel.")

        # 2. Credenciais via Vault
        token = maestro.get_credential("gabriel-telegram", "token")
        
        if not token:
            raise ValueError("Credencial do Telegram nao configurada no Vault")

        bot_tg = telebot.TeleBot(token)

        @bot_tg.message_handler(commands=['start', 'help'])
        def start(message):
            texto = (
                "Sistema Automatizado de Curadoria Cinematográfica.\n\n"
                "COMANDOS:\n"
                "/recomendar — Puxar recomendacao direta do DataPool (Maestro)\n"
                "/sugestao — Iniciar funil interativo de filtros (Q&A)"
            )
            bot_tg.reply_to(message, texto)

        @bot_tg.message_handler(commands=['recomendar'])
        def comando_recomendar(message):
            bot_tg.send_message(message.chat.id, "Consultando DataPool no Maestro...")
            try:
                task_id = execution.task_id if execution else None
                datapool = maestro.get_datapool("gabriel-filmes")
                
                if datapool.has_next():
                    entry = datapool.next(task_id=task_id)
                    filme = entry.values
                    
                    msg = formatar_mensagem_tecnica(filme, "Fila Maestro")
                    markup = teclado_pos_sugestao()
                    
                    if filme.get("poster"):
                        bot_tg.send_photo(message.chat.id, photo=filme["poster"], caption=msg, reply_markup=markup)
                    else:
                        bot_tg.send_message(message.chat.id, text=msg, reply_markup=markup)

                    entry.report_done()
                    logger.info(f"DataPool consumido: {filme.get('titulo')}")
                else:
                    bot_tg.send_message(message.chat.id, "Nao ha itens pendentes no DataPool.")
            except Exception as e:
                bot_tg.send_message(message.chat.id, f"Erro na integracao: {e}")
                logger.error(f"Erro no /recomendar: {e}")

        @bot_tg.message_handler(commands=['sugestao'])
        def comando_sugestao(message):
            estado_usuarios[message.from_user.id] = {}
            bot_tg.send_message(
                message.chat.id, 
                "Passo 1: Selecione o genero cinematografico.",
                reply_markup=gerar_teclado(GENEROS, "gen")
            )

        @bot_tg.callback_query_handler(func=lambda call: True)
        def callback_handler(call):
            user_id = call.from_user.id
            data = call.data
            chat_id = call.message.chat.id
            msg_id = call.message.message_id

            if user_id not in estado_usuarios:
                estado_usuarios[user_id] = {}

            if data.startswith("gen:"):
                estado_usuarios[user_id]["genero"] = data.split(":")[1]
                bot_tg.edit_message_text(
                    "Passo 2: Selecione o periodo de lancamento.",
                    chat_id=chat_id, message_id=msg_id,
                    reply_markup=gerar_teclado(EPOCAS, "epo")
                )

            elif data.startswith("epo:"):
                estado_usuarios[user_id]["epoca"] = data.split(":")[1]
                bot_tg.edit_message_text(
                    "Passo 3: Selecione o estilo de reconhecimento da obra.",
                    chat_id=chat_id, message_id=msg_id,
                    reply_markup=gerar_teclado(ESTILOS, "est")
                )

            elif data.startswith("est:"):
                estilo = data.split(":")[1]
                estado = estado_usuarios.get(user_id, {})
                bot_tg.edit_message_text("Filtrando base de dados local...", chat_id=chat_id, message_id=msg_id)
                
                filme = buscar_filme_local(estado.get("genero"), estado.get("epoca"), estilo)

                if not filme:
                    bot_tg.edit_message_text("Nenhuma correspondencia encontrada no catalogo.", chat_id=chat_id, message_id=msg_id)
                    return

                msg = formatar_mensagem_tecnica(filme, "QA Interativo")
                markup = teclado_pos_sugestao()
                
                if filme.get("poster"):
                    bot_tg.delete_message(chat_id, msg_id)
                    bot_tg.send_photo(chat_id, photo=filme["poster"], caption=msg, reply_markup=markup)
                else:
                    bot_tg.edit_message_text(msg, chat_id=chat_id, message_id=msg_id, reply_markup=markup)
                estado_usuarios.pop(user_id, None)

            elif data.startswith("acao:"):
                acao = data.split(":")[1]
                
                # Remove os botões da mensagem para limpar o histórico do chat
                bot_tg.edit_message_reply_markup(chat_id=chat_id, message_id=msg_id, reply_markup=None)
                
                if acao == "recomendar":
                    comando_recomendar(call.message)
                    
                elif acao == "sugestao":
                    estado_usuarios[user_id] = {}
                    bot_tg.send_message(
                        chat_id, 
                        "Passo 1: Selecione o genero cinematografico.",
                        reply_markup=gerar_teclado(GENEROS, "gen")
                    )
                    
                elif acao == "parar":
                    bot_tg.send_message(chat_id, "Sessão encerrada. Digite /start quando quiser voltar.")

        bot_tg.infinity_polling()

    except Exception as e:
        logger.error(f"Erro fatal na inicializacao: {e}")
        try:
            execution = maestro.get_execution()
            if execution:
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