# Document RAG

Production-like локальный RAG для русскоязычных PDF и TXT без LangChain.

```text
PDF/TXT → cleaning/chunking → multilingual-e5-small
        → Qdrant dense + BM25 → RRF → optional BGE reranker
        → confidence/refusal → Ollama → grounded answer + sources
```

- `fast`: reranker выключен.
- `quality`: `BAAI/bge-reranker-v2-m3`, `RERANK_CANDIDATES=15`.
- При недостаточной уверенности LLM не вызывается.
- `main.py` запускает кроссплатформенный desktop RAG Lab с подготовкой corpus,
  indexing progress, resource monitoring, benchmark artifacts и query UI.

## Установка

Требуется Python 3.11–3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Установите Ollama, загрузите text instruct модель и запустите сервер:

```bash
ollama pull qwen3:4b-instruct
ollama serve
```

Минимальная конфигурация `.env`:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct
LLM_TEMPERATURE=0
```

Остальные параметры retrieval, reranker и Qdrant перечислены в
`.env.example`. Embedded/local Qdrant предназначен для небольших smoke-тестов;
для corpus примерно от 100 MB и крупных экспериментов используйте server mode:

```bash
docker compose up -d qdrant
```

```dotenv
QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
```

## Индексация и запросы

Corpus хранится вне Git и передаётся явным путём:

```bash
python scripts/index_documents.py --path /path/to/external/corpus
python scripts/ask.py "Какая столица Литвы?" --mode fast
python scripts/ask.py "Какова максимальная глубина озера Байкал?" --mode quality
```

Desktop launcher:

```bash
python main.py
```

Подготовка небольшого Wikipedia PDF corpus из локального XML.BZ2 dump:

```bash
python scripts/build_wiki_corpus.py \
  --source data/source/ruwiki-pages-articles.xml.bz2 \
  --output data/corpus/wiki_pdf \
  --limit 100
```

Изолированный preparation benchmark, не затрагивающий baseline index:

```bash
python scripts/benchmark_ingestion.py --path data/corpus/wiki_pdf
```

Поддерживаются `.pdf` и `.txt`. Неизменившиеся файлы пропускаются по SHA-256;
PDF без извлекаемого текста не отправляются в OCR.

## API

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Какова максимальная глубина озера Байкал?","mode":"quality"}'
```

Также доступны `GET /health`, `POST /index` и Swagger UI `/docs`.

## Проверки

```bash
pytest
ruff check .
python scripts/evaluate.py --no-reranker
python scripts/evaluate.py --reranker --rerank-candidates 15
RUN_OLLAMA_INTEGRATION=1 pytest tests/test_ollama_integration.py
python scripts/compare_benchmarks.py data/benchmarks/<run-a> data/benchmarks/<run-b>
```

PDF/TXT datasets, `.env`, model weights/caches, embeddings и Qdrant indexes не
хранятся в GitHub. В репозитории остаётся только небольшой
`data/eval/retrieval_eval.json` для воспроизводимой retrieval evaluation.
