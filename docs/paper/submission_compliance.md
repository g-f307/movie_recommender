# Conformidade do pacote de submissão — ACM SAC 2027

Verificação realizada em **1º de outubro de 2026** para a issue #81. Este
documento registra as regras consultadas, o resultado das verificações e os
limites da automação. Nenhum upload foi realizado.

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

- Fonte principal: `movie_recommender_draft.tex` com
  `\documentclass[sigconf,anonymous,review,natbib=false]{acmart}`.
- Template congelado no pacote: `acmart` 2.19 e estilos bibliográficos ACM
  correspondentes, evitando dependência da versão instalada no ambiente.
- PDF: cinco páginas, tamanho US Letter (612 × 792 pt), PDF 1.7.
- Metadado `Author`: vazio; título científico preservado.
- Fontes: todas incorporadas no PDF inspecionado.
- Citações e referências cruzadas: nenhuma referência ou citação indefinida no
  log de compilação.
- Figuras: quatro PDFs vetoriais incorporados; não há rasterização de baixa
  resolução no PDF final.
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

- 15.120 células oficiais e 35 agentes sintéticos independentes;
- diferença média B5−B4 em NDCG@5 de −0,0181;
- IC bootstrap de 95% [−0,0343; −0,0022];
- Wilcoxon bicaudal p=0,0356 e correlação rank-biserial −0,406;
- 26 perdas e 9 ganhos;
- validação MovieLens 100K com 43 usuários elegíveis e empate B5−B4.

As tabelas e figuras são cópias dos artefatos rastreados produzidos na issue
#79. O empacotador calcula SHA-256 de cada arquivo e registra o commit usado.

## Auditoria das referências

Auditoria externa realizada em **1º de outubro de 2026**. As oito entradas da
bibliografia possuem DOI único, foram resolvidas pelo serviço DOI e tiveram
título, autoria, veículo, páginas e editora comparados com os metadados do
Crossref. “Acessível” significa que o identificador leva à página canônica; o
texto integral pode depender de assinatura ou acesso institucional.

| Referência | DOI | Destino canônico | Resultado |
|---|---|---|---|
| Ricci, Rokach e Shapira, *Recommender Systems Handbook* | [10.1007/978-1-4899-7637-6](https://doi.org/10.1007/978-1-4899-7637-6) | Springer | válida |
| Schein et al., *Methods and Metrics for Cold-Start Recommendations* | [10.1145/564376.564421](https://doi.org/10.1145/564376.564421) | ACM DL | válida |
| Jannach et al., *A Survey on Conversational Recommender Systems* | [10.1145/3453154](https://doi.org/10.1145/3453154) | ACM DL | válida |
| Lops, de Gemmis e Semeraro, *Content-based Recommender Systems* | [10.1007/978-0-387-85820-3_3](https://doi.org/10.1007/978-0-387-85820-3_3) | Springer | válida |
| Burke, *Hybrid Recommender Systems* | [10.1023/A:1021240730564](https://doi.org/10.1023/A:1021240730564) | Springer | válida |
| Herlocker et al., *Evaluating Collaborative Filtering Recommender Systems* | [10.1145/963770.963772](https://doi.org/10.1145/963770.963772) | ACM DL | válida |
| Castells, Hurley e Vargas, *Novelty and Diversity in Recommender Systems* | [10.1007/978-1-0716-2197-4_16](https://doi.org/10.1007/978-1-0716-2197-4_16) | Springer | válida |
| Harper e Konstan, *The MovieLens Datasets* | [10.1145/2827872](https://doi.org/10.1145/2827872) | ACM DL | válida |

O Crossref registra publicação online em 2010 para o capítulo de Lops et al. e
em 2021 para o capítulo de Castells et al. A bibliografia mantém 2011 e 2022,
respectivamente, por serem os anos editoriais das edições dos livros citadas.
Essa diferença foi revisada e não representa DOI ou obra incorreta.

## Reprodução e arquivos preparados

O comando abaixo recompila o artigo, valida o PDF e a anonimização e cria um
diretório novo sem sobrescrever um pacote existente:

```bash
make submission-package PYTHON=.venv/bin/python
```

Saída padrão, não versionada: `artifacts/submission_sac2027/`.

```text
submission_sac2027/
├── submission/
│   └── movie_recommender_anonymous.pdf
├── source/
│   ├── movie_recommender_draft.tex
│   ├── movie_recommender_references.bib
│   ├── acmart.cls e estilos bibliográficos ACM
│   └── paper_artifacts/
│       ├── figures/*.pdf
│       └── tables/*.tex
└── submission_manifest.json
```

O pacote contém somente o PDF anônimo e os fontes necessários para reprodução.
Dados, resultados privados, arquivos DVC, credenciais, `.env`, logs e
auxiliares LaTeX ficam de fora. O manifesto e o PDF devem ser regenerados no
commit aprovado imediatamente antes da submissão.

## Limites e verificação humana

A automação não substitui a conferência do formulário do sistema de submissão,
da política vigente de material suplementar nem da visualização final oferecida
pelo sistema. O autor deve repetir o checklist abaixo antes do upload. Alterar
autores, pagar taxas, aceitar cessão de direitos e efetuar o envio permanecem
fora do escopo desta issue.
