# Protocolo pré-registrado de validação externa

**Issue:** #73
**Registro:** antes do download e da análise do dataset
**Configuração:** `configs/experiments/external_validation_v1.json`

## Critérios de seleção

A fonte deve possuir usuários anônimos identificáveis, interações com timestamp, filmes com identificadores estáveis, licença de pesquisa compatível, pelo menos 100.000 avaliações e volume suficiente para B0, B4 e B5 adaptados. Fontes mutáveis de desenvolvimento não são aceitas para resultados compartilhados.

MovieLens 100K atende aos critérios e é um benchmark estável de 100.000 avaliações, aproximadamente 1.000 usuários e 1.700 filmes. O pacote é obtido exclusivamente da GroupLens. A licença exige reconhecimento, proíbe endosso implícito e uso comercial sem permissão, e não autoriza redistribuir os dados brutos sem permissão separada. Por isso o repositório versiona somente script, checksum, auditoria e resultados derivados; o arquivo bruto permanece local.

Citação obrigatória: F. Maxwell Harper e Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. DOI: 10.1145/2827872.

## Split e prevenção de leakage

As avaliações são ordenadas globalmente por timestamp, usuário e filme. Os primeiros 70% formam treino, os 10% seguintes formam adaptação e os 20% finais formam teste. B0 e os perfis estáticos são ajustados somente com treino. B5 recebe adicionalmente eventos da janela de adaptação. Nenhum evento de teste participa de popularidade, perfil, candidatos ou atualização.

Usuários elegíveis precisam ter ao menos cinco eventos em treino, um evento em adaptação e um item relevante no teste. Itens vistos em treino ou adaptação são excluídos dos candidatos desse usuário. Relevância é definida previamente como avaliação maior ou igual a quatro.

## Transportabilidade parcial

B0 representa popularidade bayesiana. B4 usa pesos de gênero derivados do histórico inicial real do usuário. B5 acrescenta evidência positiva e negativa da janela de adaptação. Essas adaptações preservam a distinção estático versus incremental, mas não reproduzem preferências declaradas, diretores, palavras-chave, personas, condições C0–C5 ou o gerador sintético.

A comparação externa testa a direção do efeito de feedback sobre NDCG@5 por usuário, não equivalência integral entre domínios. Resultados públicos e sintéticos permanecerão em tabelas e amostras separadas. Nenhum hiperparâmetro será escolhido usando o holdout sintético ou os resultados públicos.
