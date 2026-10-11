# Exposição em ciclo fechado v2

## Finalidade

A execução complementar v2 substitui a permutação neutra de itens da v1.1 por
uma trajetória causal. Em cada interação, o recomendador gera um ranking a
partir do estado disponível; o agente sintético avalia o item apresentado na
posição configurada; só então o evento é incorporado ao próximo estado. Assim,
o feedback não pode ter influenciado o ranking que o apresentou.

A mudança serve para testar B4 e B5 sob uma dinâmica de exposição mais próxima
de uma sessão de recomendação. Ela não altera a execução oficial v1.1, seus
resultados congelados ou suas conclusões.

## Contrato congelado

`configs/experiments/closed_loop_v2.yaml` define a política `closed-loop-v2`.
O arquivo usa schema estrito e lock SHA-256. Alterar qualquer decisão exige uma
nova versão, um novo lock e revisão científica; recalcular o lock não substitui
essa decisão.

| Elemento | Regra v2 |
| --- | --- |
| Métodos | B4 e B5 |
| Condições | C0=0, C1=1, C2=3, C3=5, C4=10 e C5=15 eventos |
| Seleção | primeiro item do ranking (`rank=1`) |
| Desempate | `movie_id` canônico, definido no contrato de ranking |
| Tempo lógico | dois segundos por interação |
| Esgotamento | encerra como `exhausted`, sem criar evento artificial |
| Pareamento | agente, persona, seed, condição, perfil, K e protocolo |

O item selecionado para feedback precisa pertencer ao ranking do passo anterior.
Eventos futuros e o próprio evento recém-criado são proibidos no histórico da
requisição daquele passo. Itens já consumidos deixam o conjunto candidato antes
do ranking seguinte.

## Unidade e auditoria

Uma unidade v2 é gravada por `cell_id`, e não apenas por comparação. Isso é
necessário porque B4 e B5 compartilham o mesmo estado inicial, mas B5 pode
alterar seus rankings após cada atualização. Cada arquivo preserva:

- manifesto e hash da política de exposição;
- chave de pareamento sem o método, compartilhada por B4 e B5;
- ranking, conjunto candidato, itens apresentados, evento e snapshots de cada
  interação;
- resultado da auditoria causal; e
- ranking final, relevâncias graduadas do agente e versão do estado.

A auditoria rejeita sequências fora de ordem, feedback que não foi apresentado,
histórico futuro, atualização ausente, conjunto candidato divergente, status de
falha e esgotamento inconsistente. O perfil entregue aos métodos continua sendo
a observação declarada v2, quando configurada; a preferência latente permanece
somente no agente que produz a relevância.

## Uso e limite

`configs/experiments/execution_v2.yaml` referencia tanto a observação de perfil
quanto esta política. A matriz v2 e a execução científica correspondente serão
versionadas separadamente: a geração de unidades não reutiliza o holdout v1.1
nem permite escolher retrospectivamente uma política após observar seus
resultados.

Verifique os contratos locais com:

```bash
.venv/bin/python -m unittest \
  tests.test_closed_loop_exposure \
  tests.test_execution_spec \
  tests.test_experiment_units
```

A trajetória continua sintética. Ela é evidência sobre o comportamento dos
métodos sob as políticas declaradas, não evidência de satisfação ou de efeito em
usuários humanos.
