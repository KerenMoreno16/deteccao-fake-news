# Fake News Detection - Monografia
Repositório de código da monografia - Um Estudo Comparativo de Arquiteturas de Classificação de Texto para Detecção de Fake News em Português e em Inglês.\
**Autora:** Keren Suzana B. M. Segura\
**Instituição:** Escola Politécnica da Universidade de São Paulo\
**Ano:** 2026
---
## Visão geral
Este projeto implementa e compara cinco abordagens de classificação automática de notícias falsas:
| Modelo | Paradigma |
|--------|-----------|
| Logistic Regression + TF-IDF | Baseline linear |
| MLP + TF-IDF | Baseline não-linear |
| BiLSTM + FastText | Sequencial |
| BERTimbau | Transformer PT-BR |
| BERT-base-uncased | Transformer EN |

Quatro experimentos são conduzidos:
| Exp | Descrição |
|-----|-----------|
| A | Intra-dataset (treino e teste no mesmo corpus) |
| B | Cross-dataset (zero-shot entre corpora) |
| C | Combined PT-BR / Combined EN |
| D | Split temporal (cronológico) |
---
## Datasets
| Dataset | Idioma | 
|---------|--------|
| FakeTrue.Br | PT-BR |
| Fake.br-Corpus | PT-BR |
| FakeRecogna | PT-BR |
| FakeNewsNet-PolitiFact | EN |
| CONSTRAINT-COVID19 | EN | 
| PHEME | EN |

Os arquivos de dados **não estão incluídos** neste repositório. Consulte as instruções de download em `data/README.md`.
---
## Estrutura do repositório
├── configs/
│ └── experiment.yaml # Configuração central
├── data/
│ ├── raw/ # Datasets brutos
│ └── embeddings/ # FastText .bin
├── results/
│ ├── metrics/ # JSONs por seed
│ ├── models/ # Checkpoints
│ └── predictions/ # Predições por seed
├── latex/
│ └── report_fragments/ # Tabelas e figuras
├── scripts/ # Scripts auxiliares
├── src/
│└── data/ # Loaders e splits
│└── experiments/ # Lógica por experimento
│└── features/ # TF-IDF, embeddings
│└── models/ # LR, MLP, BiLSTM, BERT
│└── utils/ # Métricas, logging
└── run_all_experiments.py
---
## Requisitos
Python 3.10 ou superior.

pip install -r requirements.txt
Para suporte a GPU (opcional):
pip install torch --index-url \

Para carregar embeddings FastText `.bin`:
pip install fasttext-wheel
---
## Reprodutibilidade
Todos os experimentos usam 10 seeds (`42` a `51`). Os resultados reportados são médias e desvios padrão amostrais (ddof=1) sobre as 10 execuções.

## Execução
### 1. Configurar ambiente
cp configs/experiment.yaml configs/local.yaml
editar configs/local.yaml com caminhos locais

### 2. Executar todos os experimentos
python scripts/run_all_experiments.py 

### 3. Gerar tabelas LaTeX
python scripts/generate_tables.py

### 4. Gerar figuras
python scripts/generate_figures.py

### 5. Verificar resultados
python scripts/run_logger.py 

---
## Resultados principais
Os resultados completos estão em `results/metrics/` após execução.
Tabelas e figuras LaTeX são geradas em `latex/report_fragments/`.
---
## Limitações conhecidas
- **PHEME**: desequilíbrio severo (~40:1) após `unverified_strategy=discard`.
  Métricas primárias: `f1_weighted` e `pr_auc`.
- **FakeNewsNet**: corpus pequeno (~1.050 artigos). IC bootstrap com cautela (n_teste ≈ 210).
- **FakeRecogna**: cobertura temporal assimétrica - fakes cobrem 2012–2021, reais apenas 2019–2021. Split temporal pode confundir período de publicação
  com veracidade.
- **Split temporal**: aplicado apenas a FakeRecogna (únicos com timestamp disponível elegível. PHEME não elegível devido desequilíbrio severo; FakeNewsNet com campo data "publish_date" em branco).
---
## Testes estatísticos
Comparações entre modelos via:
- **Wilcoxon signed-rank** (pairwise)
- **Friedman** (k ≥ 3 modelos)
- **Nemenyi post-hoc** (se Friedman significativo - Demšar, 2006)
- **Kruskal-Wallis** por dataset unidade de análise: **seeds** (n=10).
Correção de múltiplas comparações:
**Holm-Bonferroni**.
---
## Referências principais
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
- Souza et al. (2020). BERTimbau: Pretrained BERT models for Brazilian Portuguese.
- Bojanowski et al. (2017). Enriching Word Vectors with Subword Information.
- Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory*, 37(1), 145–151.
---
## Licença
Este repositório é disponibilizado para fins acadêmicos. Os datasets utilizados possuem licenças próprias - consulte cada corpus antes de uso comercial.
