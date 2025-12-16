# Stock Price Prediction using Financial News Embeddings

## Project Overview

이 repository는 The Guardian의 news articles를 사용하여 S&P 500 stock prices를 예측하는 machine learning models의 code를 포함한다. 이 project는 traditional time series features와 100명의 public figures에 대한 news article embeddings를 결합하며, multiple embedding strategies와 prediction models를 활용한다.

## Repository Contents

```
📦 Stock Price Prediction Project
├── 📂 Scraping_code          # News article scraping
├── 📂 embedding              # Text embedding generation
├── 📂 feature_datasets       # Engineered feature datasets
├── 📂 Economic_index         # S&P500 and Fear-Greed index data
├── 📂 Prediction            # Model training and evaluation
└── 📜 Feature_generation.ipynb  # Feature engineering pipeline
```

## Dataset / 데이터셋

**Download Link**: [Google Drive](https://drive.google.com/drive/folders/1mkITUxIzEL_AobnYg7SZk-DjE2Dr137Q?usp=drive_link)

Dataset은 다음을 포함한다:
- `guardian_top100_scraping.zip`: 100명의 public figures에 대한 news articles (2017-2019)
- `embedding.zip`: Multiple embedding representations
- `feature_datasets.zip`: 생성된 feature sets
---

## Pipeline 

### 1. Data Collection 

**Location**: `Scraping_code/dataset_scraping.ipynb`

#### Requirements
- `GUARDIAN_API_KEY` required (currently removed for security)

#### Process 
1. `dataset_scraping.ipynb`에서 Run all
2. Top 100 lists의 288명에 대한 articles를 scraping (2017-2019)
3. Raw data는 `guardian_raw_scraping/` folder에 저장됨
4. Filtered된 top 100 results는 `guardian_top100_scraping/`에 JSONL files로 저장됨

#### Output Format
```json
{
  "id": "uk-news/2017/...",
  "webPublicationDate": "2017-12-31T14:25:59Z",
  "headline": "New Year's ...",
  "trailText": "Tens of thousands ...",
  "bodyText": "New year celebrations are being prepared...",
  "webTitle": "New Year's Eve celebrations...",
  "webUrl": "https://www.theguardian.com/...",
  "apiUrl": "https://content.guardianapis.com/...",
  "wordcount": "402"
}
```

### 2. Text Embedding

**Location**: `embedding/` folder

네 가지 embedding strategies가 different models로 구현되었다:

| Unit | Embedding Model |
|------|----------------|
| Headlines | `BAAI/bge-large-en-v1.5` |
| Chunking + Pooling | `BAAI/bge-large-en-v1.5` |
| Full Body Text | `jinaai/jina-embeddings-v3` |
| First + Last Paragraphs | `jinaai/jina-embeddings-v3` |

#### Execution
Run each notebook: `embed_{method}.ipynb`
- `embed_headlines.ipynb`
- `embed_chunking.ipynb`
- `embed_bodyText.ipynb`
- `embed_paragraphs.ipynb`

#### Output
Each method generates:
- `vector_{method}/embeddings.npy` - Embedding vectors
- `vector_{method}/metadata.jsonl` - Metadata


### 3. Feature Engineering

**Location**: `Feature_generation.ipynb`

#### Input Data 
- S&P 500 index: `Economic_index/sp500.csv`
- Fear-Greed index: `Economic_index/fear_greed.csv`
- News embeddings: `embedding/vector_{method}/`

#### Execution
Run `Feature_generation.ipynb` to generate combined feature datasets.

#### Output
Generates multiple dataset variants in `feature_datasets/` folder:
- Format: `dataset_{feature_combination}_{embedding_method}_{pca_status}.parquet`
- Example: `dataset_D_paragraphs_pca.parquet`

Dataset variants include:
- **Dataset A**: Baseline (lag features only)
- **Dataset B-D**: Various combinations of embeddings, economic indicators, and person identifiers
- Original and PCA-reduced versions

### 4. Model Training & Prediction 

**Location**: `Prediction/` folder

#### Available Models
각 notebook은 specific model을 training하고 평가한다:

1. **Linear Regression**: `Linear regression.ipynb`
   - Results saved to: `results_lr/`
   
2. **LightGBM**: `LightGBM.ipynb`
   - Results saved to: `results_lightgbm/`
   
3. **GRU**: `GRU.ipynb`
   - Results saved to: `results_gru/`
   
4. **SARIMAX**: `SARIMAX.ipynb`
   - Results saved to: `results_sarimax/`

#### Evaluation
1. Individual model notebooks를 실행하여 predictions를 생성한다
2. `Merge_prediction.ipynb`를 실행하여 모든 results를 consolidate한다
3. 출력:
   - `evaluation_metrics.csv` - Combined evaluation metrics
   - `Results.xlsx` - Final comprehensive results

---

## Project Structure Details

```
📦 20252R0136COSE36203
├── 📂 Economic_index/              # Economic indicators
│   ├── fear_greed.csv             # Fear-Greed index
│   └── sp500.csv                   # S&P 500 historical data
│
├── 📂 Scraping_code/               # Data collection scripts
│   ├── dataset_scraping.ipynb     # Main scraping script
│   ├── people_list.txt            # Full people list
│   └── people_top100_list.txt     # Top 100 filtered list
│
├── 📂 guardian_raw_scraping/       # Raw scraped data (288 people)
│   └── {person_name}.jsonl        # Individual person's articles
│
├── 📂 guardian_top100_scraping/    # Filtered data (100 people)
│   └── {person_name}.jsonl        # Top 100 person's articles
│
├── 📂 embedding/                   # Text embedding generation
│   ├── embed_headlines.ipynb      # Headlines embedding
│   ├── embed_chunking.ipynb       # Chunking + pooling
│   ├── embed_bodyText.ipynb       # Full body text
│   ├── embed_paragraphs.ipynb     # First + last paragraphs
│   ├── vector_headlines/          # Headlines embeddings
│   ├── vector_chunking/           # Chunking embeddings
│   ├── vector_bodyText/           # Body text embeddings
│   └── vector_paragraphs/         # Paragraph embeddings
│
├── 📂 feature_datasets/            # Engineered datasets
│   ├── dataset_A.parquet          # Baseline features
│   ├── dataset_B_{method}_{pca}.parquet  # Feature set B variants
│   ├── dataset_C_{method}_{pca}.parquet  # Feature set C variants
│   └── dataset_D_{method}_{pca}.parquet  # Feature set D variants
│
├── 📂 Prediction/                  # Model training & evaluation
│   ├── Linear regression.ipynb    # Linear regression model
│   ├── LightGBM.ipynb            # LightGBM model
│   ├── GRU.ipynb                 # GRU neural network
│   ├── SARIMAX.ipynb             # SARIMAX time series model
│   ├── Merge_prediction.ipynb    # Results aggregation
│   ├── results_lr/               # Linear regression results
│   ├── results_lightgbm/         # LightGBM results
│   ├── results_gru/              # GRU results
│   ├── results_sarimax/          # SARIMAX results
│   ├── evaluation_metrics.csv    # Combined metrics
│   └── Results.xlsx              # Final results summary
│
├── Feature_generation.ipynb       # Feature engineering pipeline
└── README.md                      # This file
```

---

## Requirements

```bash
# Core libraries
numpy
pandas
torch
transformers
scikit-learn
lightgbm
statsmodels

# API & Web scraping
requests
guardian-api (with valid API key)

# Data storage
pyarrow  # for parquet files
openpyxl  # for Excel files
```

---
