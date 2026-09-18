import unittest
from cinebot_ml.experiments.ablation import AblationContext,AblationValidationError,apply_ablation,execute_ablation,load_ablation_variants

def context():
    return AblationContext("agent",42,"same-candidates",({"feedback":"like"},{"feedback":"dislike"}),{"ranked_genres":["drama"],"preferred_directors":["x"],"decade_preference":"moderno","popularity_preference":"popular","preferred_keywords":["k"]},{"genres":["drama"],"director":"x","decade":"moderno","popularity":"popular","synopsis":"s","keywords":["k"],"numeric":1})

class AblationStudyTests(unittest.TestCase):
 def test_variantes_removem_somente_componente_declarado(self):
  variants,_=load_ablation_variants(); self.assertEqual([v.name for v in variants],[f"A{i}" for i in range(7)])
  original=context(); transformed={v.name:apply_ablation(original,v) for v in variants}
  self.assertEqual(transformed["A0"],original); self.assertEqual(transformed["A1"].history,())
  self.assertEqual(len(transformed["A2"].history),1); self.assertNotIn("ranked_genres",transformed["A3"].profile)
  self.assertNotIn("director",transformed["A4"].item_features); self.assertNotIn("decade_preference",transformed["A5"].profile)
  self.assertNotIn("synopsis",transformed["A6"].item_features)
  self.assertTrue(all(item.item_features.get("numeric")==1 for item in transformed.values()))
 def test_execucao_pareada_preserva_agente_seed_candidatos_e_id(self):
  runner=lambda variant,current:{"ndcg_at_k":0.5,"related_metric":len(current.history)}
  first=execute_ablation(context(),runner); second=execute_ablation(context(),runner)
  self.assertEqual(first.study_id,second.study_id); self.assertEqual({r["agent_id"] for r in first.rows},{"agent"})
  self.assertEqual({r["candidate_set_id"] for r in first.rows},{"same-candidates"}); self.assertTrue(all("active_components" in r for r in first.rows))
 def test_runner_sem_metrica_primaria_falha(self):
  with self.assertRaisesRegex(AblationValidationError,"ndcg"):
   execute_ablation(context(),lambda variant,current:{})

if __name__=="__main__": unittest.main()
