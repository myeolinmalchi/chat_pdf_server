# Embed datasets and QA with it.

## How to Run

1. Clone repository and move directory.

```
git clone https://github.com/myeolinmalchi/chat_pdf_server.git
cd chat_pdf_server
```

2. Install dependencies.

```
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`.

```dosini
OPENAI_API_KEY=your_openai_api_key
CHROMA_DB_HOST=chroma_db_server_host
CHROMA_DB_PORT=chroma_db_server_port
CHROMA_DB_COLLECTION=chroma_db_collection
```

4. Create `dataset/` directory and move `.pdf` files.

```
mkdir dataset
```

5. Run Scripts.

```
python embed_dataset.py
python -m uvicorn main:app --reload
```

Now, Server is run on `127.0.0.1:8000`.

You can find API documentation at `127.0.0.1:8000/docs`.
