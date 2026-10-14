# Conformidade do pacote de submissão — ACM SAC 2027

Verificação original realizada em **1º de outubro de 2026** e atualizada em
**13 de outubro de 2026** após a revisão metodológica e editorial do manuscrito.
Este documento registra as regras consultadas, o resultado das verificações e
os limites da automação. Nenhum upload foi realizado.

## Regras oficiais registradas

| Item | Regra adotada | Fonte oficial |
|---|---|---|
| Evento | 42nd ACM/SIGAPP Symposium on Applied Computing, 5–9 de abril de 2027, Gwangju, Coreia do Sul | [SAC 2027](https://www.sigapp.org/sac/sac2027/) |
| Categoria | manuscrito regular; não há submissão separada para pôster | [Submission](https://www.sigapp.org/sac/sac2027/submission.php) |
| Trilha | SEAI — *Smarter Engineering: Building AI and Building with AI* | [Tracks](https://www.sigapp.org/sac/sac2027/tracks.php) e [SEAI](https://tziadi.github.io/seai-sac2027/) |
| Prazo | **16 de outubro de 2026, EST**, conforme prorrogação publicada no site geral em 30/09/2026 | [SAC 2027](https://www.sigapp.org/sac/sac2027/) |
| Revisão | duplo-cega, por pelo menos três revisores | [Submission](https://www.sigapp.org/sac/sac2027/submission.php) |
| Extensão | até oito páginas; o author kit informa possibilidade de até duas páginas adicionais | [Author kit](https://www.sigapp.org/sac/sac2027/authorkit.php) |
| Formato | template ACM, classe `acmart`, formato `sigconf` | [Author kit](https://www.sigapp.org/sac/sac2027/authorkit.php) |

A página própria da SEAI ainda exibia 2 de outubro quando consultada. Adotou-se
a atualização posterior e geral da conferência, de 30/09/2026, que prorrogou o
prazo para 16/10/2026. Essa divergência deve ser conferida novamente no sistema
de submissão imediatamente antes do upload.

O trabalho se enquadra na SEAI como estudo empírico e infraestrutura
reproduzível para teste e validação de um sistema inteligente. O manuscrito não
faz alegações sobre participantes humanos.

## Resultado técnico

- Fonte principal: `movie_recommender_sac2027_en.tex` com
  `\documentclass[sigconf,anonymous,review,natbib=false]{acmart}`.
- Template congelado no pacote: `acmart` 2.19 e estilos bibliográficos ACM
  correspondentes, evitando dependência da versão instalada no ambiente.
- PDF: oito páginas, tamanho US Letter (612 × 792 pt); o corpo termina na
  página 7 e as 30 referências ocupam somente a página 8.
- Metadado `Author`: vazio; título científico preservado.
- Fontes: todas incorporadas no PDF inspecionado.
- Citações e referências cruzadas: nenhuma referência ou citação indefinida no
  log de compilação.
- Figuras: quatro PDFs vetoriais incorporados, ajustados à largura de uma
  coluna e com descrição alternativa; não há rasterização de baixa resolução.
- Tabelas: inspecionadas no PDF em coluna; nenhuma ultrapassa a área útil.
- Log: há avisos `Underfull \vbox` e o aviso da classe sobre o bloco de
  referência ACM suprimido no modo de revisão. Não há `Overfull`, erro de
  compilação ou elemento cortado.
- Anonimização automatizada: nenhum nome conhecido, email, caminho local,
  username, variável de credencial ou endpoint privado encontrado no fonte e
  no texto extraído do PDF.
- Autocitações: a bibliografia não identifica os autores do manuscrito e o
  texto não aponta para o repositório privado.

## Consistência científica

Os números centrais foram mantidos sem alteração e permanecem vinculados às
fontes estruturadas por [`manuscript_traceability.md`](manuscript_traceability.md):

- 15.120 células e 35 trajetórias sintéticas aninhadas em sete personas;
- diferença média B5−B4 em NDCG@5 de −0,0181;
- IC bootstrap de 95% [−0,0343; −0,0022];
- Wilcoxon bicaudal p=0,0356 e correlação rank-biserial −0,406;
- 26 perdas e 9 ganhos;
- validação MovieLens 100K com 43 usuários elegíveis e empate B5−B4.

As tabelas e figuras são cópias dos artefatos rastreados produzidos na issue
#79. O empacotador calcula SHA-256 de cada arquivo e registra o commit usado.

## Auditoria das referências

A bibliografia atual contém 30 entradas. Em **13 de outubro de 2026**, os 25
DOIs foram resolvidos por `doi.org`; códigos 403 ocorreram apenas no destino de
editoras que bloqueiam consultas automatizadas, depois da resolução correta do
identificador. As duas entradas arXiv usam identificadores canônicos sem repetir
o mesmo endereço nos campos bibliográficos. Os trabalhos de
Schnabel et al., Rocchio e Holm não dependem de DOI no manuscrito e mantêm os
dados editoriais de seus veículos originais. A compilação com Biber não reporta
chaves ausentes ou entradas inválidas.

O Crossref registra publicação online em 2010 para o capítulo de Lops et al. e
em 2021 para o capítulo de Castells et al. A bibliografia mantém 2011 e 2022,
respectivamente, por serem os anos editoriais das edições dos livros citadas.
Essa diferença foi revisada e não representa DOI ou obra incorreta.

## Reprodução e arquivos preparados

O pacote local abaixo contém a versão inglesa revisada no formato ACM, a
bibliografia e as quatro figuras usadas no manuscrito. Ele permanece fora do
versionamento para não misturar arquivos de submissão com o código-fonte:

```bash
cd submission/prism_sac2027
latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir=build main.tex
```

Diretório não versionado: `submission/prism_sac2027/`.

```text
prism_sac2027/
├── main.tex
├── main.bbl
├── references.bib
├── acmart.cls e estilos bibliográficos ACM
├── figures/
│   ├── study_design_column.pdf
│   ├── evidence_lines_column.pdf
│   ├── ablation_column.pdf
│   └── robustness_column.pdf
└── build/main.pdf
```

Esse diretório é o pacote de compilação do artigo, não o artefato experimental.
O suplemento anônimo com código, configurações e manifestos ainda deve ser
publicado ou anexado no portal de submissão. Dados privados, credenciais,
`.env`, endpoints e caminhos locais não devem integrar nenhum dos pacotes.

## Limites e verificação humana

A automação não substitui a conferência do formulário do sistema de submissão,
da política vigente de material suplementar nem da visualização final oferecida
pelo sistema. O autor deve repetir o checklist abaixo antes do upload. Alterar
autores, pagar taxas, aceitar cessão de direitos e efetuar o envio permanecem
fora do escopo desta issue.
