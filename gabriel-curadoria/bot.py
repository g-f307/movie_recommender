import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from botcity.maestro import BotMaestroSDK, AutomationTaskFinishStatus, DataPoolEntry

# Helpers 
def carregar_filmes(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Entry point 
def main():
    load_dotenv()
    
    # 1. Conexão híbrida (funciona local ou no Maestro)
    try:
        maestro = BotMaestroSDK.from_sys_args()
    except Exception:
        maestro = BotMaestroSDK(
            server=os.getenv("MAESTRO_SERVER"),
            login=os.getenv("MAESTRO_LOGIN"),
            key=os.getenv("MAESTRO_KEY"),
        )
        maestro.login(
            server=os.getenv("MAESTRO_SERVER"),
            login=os.getenv("MAESTRO_LOGIN"),
            key=os.getenv("MAESTRO_KEY"),
        )

    print(f"[curadoria] Iniciando processamento — {datetime.now()}")

    try:
        execution = maestro.get_execution()
        # Se rodar via Maestro, tenta pegar o caminho via parâmetro. Se local, usa a pasta padrão.
        data_path_str = execution.parameters.get("DATA_PATH") if execution else "data/filmes.json"
        DATA_PATH = Path(data_path_str)

        # 2. Carrega os dados brutos do Scraper
        dados = carregar_filmes(DATA_PATH)
        if not dados or "perfis" not in dados:
            raise ValueError("Arquivo filmes.json nao encontrado ou vazio. Rode o Scraper primeiro.")

        # 3. Conecta no DataPool
        datapool = maestro.get_datapool("gabriel-filmes")

        # 4. Controle de Histórico (Garante que filmes não sejam duplicados)
        historico_path = Path("historico_inseridos.json")
        inseridos = []
        if historico_path.exists():
            with open(historico_path, "r", encoding="utf-8") as f:
                inseridos = json.load(f)

        novos_filmes = 0

        # 5. Varre os filmes raspados e insere os novos no DataPool
        for perfil, filmes in dados["perfis"].items():
            for filme in filmes:
                if filme["id"] not in inseridos:
                    
                    # Formata os dados para as colunas exatas do DataPool
                    entry = DataPoolEntry(values={
                        "id":            str(filme["id"]),
                        "titulo":        filme["titulo"],
                        "sinopse":       filme.get("sinopse", ""),
                        "ano":           str(filme.get("ano", "")),
                        "duracao":       str(filme.get("duracao", "")),
                        "diretor":       filme.get("diretor", ""),
                        "nacionalidade": filme.get("nacionalidade", ""),
                        "streaming":     ", ".join(filme.get("streaming", [])),
                        "nota":          str(filme.get("nota", "")),
                        "votos":         str(filme.get("votos", "")),
                        "url":           filme["url"],
                        "poster":        filme.get("poster", ""),
                        "perfil":        filme.get("perfil", ""),
                        "genero":        filme.get("genero", ""),
                    })
                    
                    datapool.create_entry(entry)
                    inseridos.append(filme["id"])
                    novos_filmes += 1
                    print(f"[curadoria] Adicionado ao DataPool: {filme['titulo']}")

        # 6. Salva o histórico atualizado
        historico_path.parent.mkdir(parents=True, exist_ok=True)
        with open(historico_path, "w", encoding="utf-8") as f:
            json.dump(inseridos, f, indent=2)

        # 7. Envia o histórico como artefato e finaliza a tarefa
        if execution:
            maestro.post_artifact(
                task_id=execution.task_id,
                artifact_name="historico_inseridos.json",
                filepath=str(historico_path),
            )
            maestro.finish_task(
                task_id=execution.task_id,
                status=AutomationTaskFinishStatus.SUCCESS,
                message=f"Curadoria concluida. {novos_filmes} novos filmes inseridos no DataPool.",
            )
        else:
            print(f"[curadoria] Finalizado localmente. {novos_filmes} novos filmes inseridos.")

    except Exception as e:
        print(f"[curadoria] Erro fatal: {e}")
        try:
            if 'execution' in locals() and execution:
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