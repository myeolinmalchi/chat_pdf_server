import os
from chromadb import Settings
import chromadb

from dotenv import load_dotenv
from db import SessionLocal, get_password_hash, User

load_dotenv()

##################################################################

INITIAL_USER_NAME = os.environ['INITIAL_USER_NAME']
INITIAL_USER_PASSWORD = os.environ['INITIAL_USER_PASSWORD']

db = SessionLocal()

db_user = User(
    username=INITIAL_USER_NAME,
    password=get_password_hash(INITIAL_USER_PASSWORD)
)

db.add(db_user)

db.commit()

db.close()

##################################################################

CHROMA_DB_HOST = os.environ['CHROMA_DB_HOST']
CHROMA_DB_PORT = os.environ['CHROMA_DB_PORT']
COLLECTION_NAME = os.environ['CHROMA_DB_COLLECTION']

client_settings = Settings(
    chroma_api_impl="rest", 
    chroma_server_host=CHROMA_DB_HOST, 
    chroma_server_http_port=CHROMA_DB_PORT
)

client = chromadb.Client(settings = client_settings)

if COLLECTION_NAME not in [collection.name for collection in client.list_collections()]:
    client.create_collection(COLLECTION_NAME)

##################################################################
