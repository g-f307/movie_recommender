# Agentes sintéticos heterogêneos

Os agentes sintéticos constituem a fonte controlada de relevância da simulação.
Cada agente recebe uma preferência latente antes de qualquer recomendação e
avalia filmes sem conhecer o método, o ranking, a posição ou o score produzido
pelo recomendador. Essa separação evita favorecer artificialmente um baseline.

## Configuração congelada

A população v1 está definida em `configs/agents/personas_v1.yaml`, validada por
JSON Schema e protegida por lock SHA-256. Seus parâmetros foram definidos *a
priori*, sem consulta ao holdout. Mudanças de comportamento ou distribuição
devem criar uma nova versão, sem recalcular o lock da v1 após observar os
resultados.

A coorte oficial contém 140 agentes, igualmente distribuídos entre sete
personas:

| Persona | Comportamento representado |
| --- | --- |
| `consistent` | resposta estável e baixa temperatura |
| `noisy` | maior variabilidade na utilidade e na resposta |
| `exploratory` | chance explícita de resposta não guiada pela preferência |
| `specific` | concentração em um único gênero |
| `broad` | preferência distribuída por até seis gêneros disponíveis |
| `popularity_sensitive` | maior peso para a popularidade do item |
| `contradictory` | inversões controladas entre utilidade e feedback |

Cada instância possui `agent_id`, seed e `preference_id` estáveis. A distribuição
da coorte, os parâmetros e o hash da configuração podem ser registrados no
manifesto experimental.

## Preferência e utilidade latentes

Gêneros, diretores, palavras-chave, década e sensibilidade à popularidade são
amostrados deterministicamente a partir da versão, persona, seed e índice do
agente. O catálogo fornece apenas o vocabulário possível; nenhum resultado de
recomendação participa dessa construção.

A utilidade de um filme combina aderência de gênero, diretor, palavras-chave,
década e popularidade, além de gosto latente por item e ruído da persona. O
valor final é limitado ao intervalo `[0,1]`. Essa função é uma hipótese
geradora de dados, não uma medida observada de satisfação humana.

## Política de resposta

A utilidade é transformada em probabilidade de like por uma função logística.
Temperatura, exploração e contradição controlam comportamentos distintos. Uma
amostragem determinística por agente, filme, interação e canal produz
`like`/`dislike`; a mesma avaliação também informa relevância graduada de zero
a três e identifica sua origem como `synthetic_user`.

Repetir a mesma configuração, catálogo e seed reproduz preferências e
respostas. Alterar a seed gera outra trajetória. A API pública de avaliação
recebe somente o filme e o índice lógico da interação, tornando a independência
do recomendador verificável por teste.

## Uso

```python
from cinebot_ml.simulation import build_agent, build_agent_cohort

agent = build_agent("consistent", seed=42, catalog=catalog)
judgment = agent.evaluate(movie, interaction=0)

cohort = build_agent_cohort(catalog, base_seed=42)
manifest = agent.manifest()
```

## Limitações e validade

Os agentes ampliam escala, diversidade controlada e reprodutibilidade, mas não
substituem evidência humana. Os resultados demonstram comportamento do sistema
sob o modelo gerador especificado. Portanto, o artigo deve reportar parâmetros,
seeds e análises de sensibilidade, distinguir feedback sintético do real e
evitar generalizações sobre preferência humana ou impacto em produção.
