import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cinebot_ml.dataset import movie_genres
from cinebot_ml.simulation import AgentConfigError, SyntheticAgent, build_agent, build_agent_cohort, load_agent_config
from cinebot_ml.simulation.agents import DEFAULT_AGENT_CONFIG_PATH


def catalog():
    genres = ["drama", "terror", "comedia", "acao", "ficcao", "romance", "documentario"]
    return [{"id": i, "genero": genre, "perfil": genre, "generos_secundarios": [genres[(i+1)%len(genres)]], "ano": 1980 + i*5, "votos": i*10000, "diretor": f"Diretor {i%4}", "palavras_chave": [f"tema {i%5}"]} for i, genre in enumerate(genres, 1)]


class SyntheticAgentTests(unittest.TestCase):
    def test_sete_comportamentos_estao_configurados(self):
        config = load_agent_config()
        self.assertEqual(set(config.personas), {"consistent", "noisy", "exploratory", "specific", "broad", "popularity_sensitive", "contradictory"})

    def test_mesma_persona_seed_e_catalogo_produzem_mesmo_agente(self):
        first = build_agent("consistent", 42, catalog())
        second = build_agent("consistent", 42, catalog())
        self.assertEqual(first.preference, second.preference)
        self.assertEqual(first.manifest(), second.manifest())

    def test_seeds_diferentes_produzem_preferencias_diferentes(self):
        self.assertNotEqual(build_agent("broad", 42, catalog()).preference_id, build_agent("broad", 43, catalog()).preference_id)

    def test_agente_gera_like_dislike_e_relevancia_graduada(self):
        agent = build_agent("noisy", 42, catalog())
        judgments = [agent.evaluate(movie, interaction) for interaction in range(10) for movie in catalog()]
        self.assertEqual({item.source for item in judgments}, {"synthetic_user"})
        self.assertEqual({item.feedback for item in judgments}, {"like", "dislike"})
        self.assertTrue(all(0 <= item.graded_relevance <= 3 for item in judgments))

    def test_consistente_e_reproduzivel(self):
        agent = build_agent("consistent", 42, catalog())
        self.assertEqual(agent.evaluate(catalog()[0], 1), agent.evaluate(catalog()[0], 1))

    def test_ruido_e_exploracao_estao_explicitos(self):
        noisy = build_agent("noisy", 42, catalog())
        exploratory = build_agent("exploratory", 42, catalog())
        self.assertGreater(noisy.parameters["noise"], 0)
        self.assertGreater(exploratory.parameters["exploration"], 0)

    def test_preferencia_especifica_e_ampla(self):
        self.assertEqual(len(build_agent("specific", 42, catalog()).preference.genres), 1)
        broad = build_agent("broad", 42, catalog())
        available_genres = {
            genre for movie in catalog() for genre in movie_genres(movie)
        }
        self.assertEqual(
            len(broad.preference.genres),
            min(6, len(available_genres)),
        )

    def test_sensibilidade_a_popularidade(self):
        agent = build_agent("popularity_sensitive", 42, catalog())
        item = dict(catalog()[0]); item["id"] = 999; item["votos"] = 0
        low = agent.utility(item, 0); item["votos"] = 1_000_000
        self.assertGreater(agent.utility(item, 0), low)

    def test_contradicao_controlada_inverte_probabilidade(self):
        base = build_agent("consistent", 42, catalog())
        normal = SyntheticAgent(base.agent_id, "normal", base.seed, base.preference, {**base.parameters, "contradiction": 0.0}, base.config)
        contrary = SyntheticAgent(base.agent_id, "contrary", base.seed, base.preference, {**base.parameters, "contradiction": 1.0}, base.config)
        self.assertAlmostEqual(normal.evaluate(catalog()[0], 0).like_probability + contrary.evaluate(catalog()[0], 0).like_probability, 1.0)

    def test_api_nao_recebe_metodo_ranking_ou_score(self):
        parameters = set(inspect.signature(SyntheticAgent.evaluate).parameters)
        self.assertEqual(parameters, {"self", "movie", "interaction"})

    def test_probabilidades_sao_finitas_e_limitadas(self):
        for persona in load_agent_config().personas:
            judgment = build_agent(persona, 42, catalog()).evaluate(catalog()[0], 0)
            self.assertTrue(0 <= judgment.utility <= 1)
            self.assertTrue(0 <= judgment.like_probability <= 1)

    def test_cohort_registra_quantidade_e_distribuicao(self):
        config = load_agent_config(); agents = build_agent_cohort(catalog(), 42, config)
        self.assertEqual(len(agents), sum(config.cohort.values()))
        self.assertEqual(len({agent.agent_id for agent in agents}), len(agents))

    def test_configuracao_invalida_e_lock_sao_bloqueados(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "personas_v1.yaml"
            payload = yaml.safe_load(DEFAULT_AGENT_CONFIG_PATH.read_text())
            payload["personas"]["noisy"]["noise"] = 2
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
            digest = hashlib.sha256(path.read_bytes()).hexdigest(); path.with_suffix(".lock.json").write_text(json.dumps({"sha256": digest}))
            with self.assertRaisesRegex(AgentConfigError, "inválida"): load_agent_config(path)
            path.write_bytes(DEFAULT_AGENT_CONFIG_PATH.read_bytes()); path.with_suffix(".lock.json").write_text(json.dumps({"sha256": "0"*64}))
            with self.assertRaisesRegex(AgentConfigError, "alterada"): load_agent_config(path)


if __name__ == "__main__": unittest.main()
