import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from botcity.maestro import AutomationTaskFinishStatus, BotMaestroSDK, DataPoolEntry


def carregar_filmes(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def obter_execucao(maestro: BotMaestroSDK | None):
    if maestro is None:
        return None
    try:
        return maestro.get_execution()
    except Exception:
        return None


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "sim"}


def carregar_maestro() -> BotMaestroSDK | None:
    try:
        maestro = BotMaestroSDK.from_sys_args()
        if maestro.server:
            return maestro
    except Exception:
        pass

    if not parse_bool(os.getenv("USE_DATAPOOL"), default=False):
        return None

    maestro = BotMaestroSDK()

    maestro.login(
        server=os.getenv("MAESTRO_SERVER"),
        login=os.getenv("MAESTRO_LOGIN"),
        key=os.getenv("MAESTRO_KEY"),
    )
    return maestro


def carregar_lista_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def salvar_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_datapool_entry(filme: dict) -> DataPoolEntry:
    return DataPoolEntry(
        values={
            "id": str(filme["id"]),
            "titulo": filme["titulo"],
            "sinopse": filme.get("sinopse", ""),
            "ano": str(filme.get("ano", "")),
            "duracao": str(filme.get("duracao", "")),
            "diretor": filme.get("diretor", ""),
            "nacionalidade": filme.get("nacionalidade", ""),
            "streaming": ", ".join(filme.get("streaming", [])),
            "nota": str(filme.get("nota", "")),
            "votos": str(filme.get("votos", "")),
            "url": filme["url"],
            "poster": filme.get("poster", ""),
            "perfil": filme.get("perfil", ""),
            "genero": filme.get("genero", ""),
        }
    )


def main() -> None:
    load_dotenv()
    maestro = carregar_maestro()

    print(f"[curadoria] Iniciando processamento — {datetime.now()}")

    try:
        execution = obter_execucao(maestro)
        parameters = execution.parameters if execution else {}

        data_path = parameters.get("DATA_PATH") or os.getenv("DATA_PATH", "data/filmes.json")
        queue_path = Path(
            parameters.get("LOCAL_QUEUE_PATH")
            or os.getenv("LOCAL_QUEUE_PATH", "data/fila_curadoria.json")
        )
        historico_path = Path(
            parameters.get("CURADORIA_HISTORY_PATH")
            or os.getenv("CURADORIA_HISTORY_PATH", "data/historico_inseridos.json")
        )
        dry_run = parse_bool(
            parameters.get("CURADORIA_DRY_RUN") or os.getenv("CURADORIA_DRY_RUN"),
            default=False,
        )
        use_datapool = parse_bool(
            parameters.get("USE_DATAPOOL") or os.getenv("USE_DATAPOOL"),
            default=False,
        )
        data_file = Path(data_path).expanduser()

        existe_arquivo = data_file.exists()
        tamanho_arquivo = data_file.stat().st_size if existe_arquivo else 0
        print(
            f"[curadoria] DATA_PATH={data_file} | existe={existe_arquivo} | tamanho={tamanho_arquivo} bytes"
        )

        dados = carregar_filmes(data_file)
        if not dados or "perfis" not in dados:
            raise ValueError("Arquivo filmes.json não encontrado ou vazio. Rode o Scraper primeiro.")

        inseridos = carregar_lista_json(historico_path)
        fila_local = carregar_lista_json(queue_path) if queue_path.exists() else []
        ids_em_fila = {item["id"] for item in fila_local if isinstance(item, dict) and "id" in item}
        novos_filmes = 0
        datapool = maestro.get_datapool("gabriel-filmes") if use_datapool and maestro else None

        for _, filmes in dados["perfis"].items():
            for filme in filmes:
                if filme["id"] in inseridos:
                    continue

                if use_datapool and datapool is not None:
                    if not dry_run:
                        datapool.create_entry(build_datapool_entry(filme))
                    acao = "Simulado no DataPool" if dry_run else "Adicionado ao DataPool"
                else:
                    if filme["id"] not in ids_em_fila:
                        fila_local.append(filme)
                        ids_em_fila.add(filme["id"])
                    acao = "Adicionado à fila local"

                inseridos.append(filme["id"])
                novos_filmes += 1
                print(f"[curadoria] {acao}: {filme['titulo']}")

        if not dry_run:
            salvar_json(inseridos, historico_path)
            if not use_datapool:
                salvar_json(fila_local, queue_path)

        if dry_run:
            destino = "DataPool" if use_datapool else "fila local"
            mensagem_final = f"Curadoria concluída em modo dry run. {novos_filmes} filmes seriam enviados para {destino}."
            print(f"[curadoria] Dry run concluído. {novos_filmes} filmes seriam enviados para {destino}.")
        elif use_datapool:
            mensagem_final = f"Curadoria concluída. {novos_filmes} novos filmes enviados ao DataPool."
            print(f"[curadoria] Finalizado localmente. {novos_filmes} novos filmes enviados ao DataPool.")
        else:
            mensagem_final = f"Curadoria concluída. {novos_filmes} filmes adicionados em {queue_path}."
            print(f"[curadoria] Finalizado localmente. {novos_filmes} filmes adicionados em {queue_path}.")

        if execution and maestro:
            if historico_path.exists():
                maestro.post_artifact(
                    task_id=execution.task_id,
                    artifact_name="historico_inseridos.json",
                    filepath=str(historico_path),
                )
            if not use_datapool and queue_path.exists():
                maestro.post_artifact(
                    task_id=execution.task_id,
                    artifact_name="fila_curadoria.json",
                    filepath=str(queue_path),
                )
            maestro.finish_task(
                task_id=execution.task_id,
                status=AutomationTaskFinishStatus.SUCCESS,
                message=mensagem_final,
            )

    except Exception as exc:
        print(f"[curadoria] Erro fatal: {exc}")
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
