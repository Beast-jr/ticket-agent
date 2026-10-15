"""Day 1 check: can we reach Postgres, and is the pgvector extension available?"""
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()  # reads DATABASE_URL from the .env file

url = os.environ["DATABASE_URL"]

with psycopg.connect(url, autocommit=True) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("pgvector ready")
