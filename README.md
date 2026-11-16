# ЛР10 — Metal-Detail
Навчальний пакет: синтетичний корпус 20 документів/200 фрагментів; chunking; FAISS HNSW; дві multilingual embeddings моделі; BM25+dense+RRF(k=60)+cross-encoder; gold 40 запитів.

## Установлення
`python -m venv .venv` і активуйте середовище, потім `pip install -r requirements.txt`.

## Запуск з кореня проєкту
`python src/chunking.py`
`python src/index.py --model e5`
`python src/index.py --model minilm`
`python src/evaluate.py`
`python src/benchmark.py --model e5`
`python src/hybrid.py --query "Як перевірити заготовку зі сталі 45?"`

Моделі завантажуються під час першого запуску; потрібен доступ до Hugging Face. Час/метрики залежать від комп’ютера. Перед здачею замініть демонстраційні оцінки другого анотатора незалежною розміткою.
