# Checklist final de submissão — ACM SAC 2027 / SEAI

Data da verificação técnica: **2026-10-13**

Responsável técnico: pipeline reproduzível da issue #81

Estado: **pronto para revisão final do autor; não submetido**

## Conformidade e anonimização

- [x] Categoria, trilha, prazo, limite e fontes oficiais registrados.
- [x] Classe `acmart` configurada como `sigconf,anonymous,review`.
- [x] PDF com oito páginas em US Letter; corpo nas páginas 1--7 e referências na página 8.
- [x] Nomes, afiliações, emails e agradecimentos ausentes.
- [x] Campo `Author` dos metadados vazio.
- [x] Autocitações e menções ao repositório revisadas.
- [x] Caminhos locais, usernames, credenciais e endpoints privados ausentes.
- [x] Fontes incorporadas; imagens vetoriais e tabelas legíveis.
- [x] Citações, referências e legendas resolvidas.
- [x] Trinta referências compiladas; 25 DOIs resolvidos e URLs arXiv/PMLR/JSTOR normalizados.
- [x] Rótulos, eixos, legendas e tabelas editoriais integralmente em inglês.

## Integridade e reprodução

- [x] Manuscrito compilado com `latexmk -halt-on-error`.
- [x] Números centrais conferidos contra a rastreabilidade congelada.
- [x] Suíte automatizada e `git diff --check` executados.
- [x] Pacote mínimo produzido sem dados ou artefatos privados.
- [x] Manifesto contém SHA-256, commit, data e inventário relativo.
- [x] Empacotador recusa sobrescrever diretório não vazio.
- [x] Nenhum upload foi realizado automaticamente.

## Conferência imediatamente antes do upload

- [ ] Confirmar no portal que a prorrogação para 16/10/2026 se aplica à SEAI.
- [ ] Publicar ou anexar o suplemento anônimo com código, configurações e manifestos.
- [ ] Inserir no portal e, se permitido, no manuscrito o identificador anônimo do suplemento.
- [ ] Regenerar o pacote do artigo a partir do commit aprovado e guardar o log completo.
- [ ] Abrir o PDF enviado pelo próprio portal e revisar as oito páginas.
- [ ] Comparar o SHA-256 enviado com o PDF local aprovado.
- [ ] Preencher autores, ORCIDs e conflitos apenas nos campos privados do portal.
- [ ] Confirmar a política vigente para fonte ou material suplementar.
- [ ] Obter autorização explícita do autor responsável e somente então enviar.

Assinatura técnica: **movie_recommender / issue #81**  
Assinatura do autor responsável antes do upload: __________________________
