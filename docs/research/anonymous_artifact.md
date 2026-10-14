# Artefato anônimo de reanálise — SAC 2027

## Finalidade

O artefato permite conferir a integridade das evidências e refazer a análise
estatística e as figuras da execução v2.1 sem acessar o remoto DVC, o Cloudflare
R2, credenciais ou variáveis do `.env`.

O pacote oferece **verificação e reanálise**, não uma reprodução integral desde
o catálogo. O catálogo original, tabelas cartesianas de treinamento e feedback
real permanecem excluídos por política de distribuição e privacidade.

## Conteúdo permitido

- manifesto final e 20.160 células v2.1;
- seleção registrada na validação e suas observações;
- resultados agregados e figuras v2.1;
- evidências derivadas v1.1 de contrastes, ablação, robustez, descoberta e
  validação externa;
- configurações e locks experimentais;
- código necessário para a análise;
- dependências, licença anônima de revisão, instruções e verificador offline;
- inventário de tamanho e SHA-256 por arquivo.

O exportador usa uma lista permitida. `data/`, `datasets/`, `.env`, cache DVC,
histórico Git, pacote do manuscrito e módulos com identidade do autor não entram
no arquivo.

Para compatibilidade com o Anonymous GitHub, nenhum membro pode exceder 8 MiB.
O `matrix.manifest.json` é armazenado como `matrix.manifest.json.gz`; a análise
descompacta o conteúdo e confere o hash do JSON original antes de utilizá-lo.

## Geração

```bash
make anonymous-artifact PYTHON=.venv/bin/python
```

Saída local ignorada pelo Git:

```text
artifacts/anonymous_sac2027_reanalysis.zip
```

Um caminho alternativo pode ser informado sem alterar o código:

```bash
OUTPUT_PATH=/tmp/anonymous_sac2027.zip \
  make anonymous-artifact PYTHON=.venv/bin/python
```

O comando interrompe a geração se detectar email, caminho de usuário, endpoint
privado, marcador de credencial ou identificador proibido em conteúdo textual.
O arquivo é escrito primeiro em nome temporário e só passa ao nome final depois
de concluído.

## Verificação independente

Após extrair o ZIP:

```bash
python verify_artifact.py
python -m venv .venv
.venv/bin/python -m pip install -r requirements-ci.txt
.venv/bin/python -m cinebot_ml.analysis.v2_analysis \
  --manifest results/v2_1/revision_2/manifests/executions/28732d8425f71f151655617c/execution.manifest.json \
  --selection results/v2_1/revision_2/reports/validation/0b1bcc4e1bd908b021949dea/selection.json \
  --output-dir reproduced/v2_analysis
```

O diretório `reproduced/v2_analysis` deve produzir novamente o relatório JSON,
as linhas pareadas, os subgrupos, as figuras e um manifesto de saída. Os hashes
dessas saídas podem ser comparados com o diretório de análise incluído.

## Publicação

Antes do upload em serviço anônimo:

1. extrair em diretório limpo;
2. executar `verify_artifact.py`;
3. executar a reanálise;
4. comparar os hashes das saídas;
5. abrir o pacote sem autenticação em uma sessão privada;
6. confirmar que título, descrição e URL não revelam autoria ou instituição.

Na publicação por repositório, o ZIP deve ser extraído em um repositório privado
dedicado. O arquivo ZIP não deve ser versionado, pois o serviço opera sobre os
arquivos individuais do repositório.

O endereço anônimo deve ser acrescentado ao manuscrito somente depois dessas
verificações e sem prometer reconstrução dos dados restritos.

## Autorização para revisão

O titular do projeto autorizou a hospedagem temporária de uma cópia anonimizada,
o acesso pelos revisores e a reprodução local necessária para verificar os
resultados. O pacote inclui `LICENSE-REVIEW.md`, que registra essa permissão sem
revelar a identidade do autor. A licença identificada da raiz não entra no ZIP.

Essa autorização é limitada à revisão científica. Ela não converte o projeto
em software de código aberto nem permite redistribuição fora do processo de
avaliação.
