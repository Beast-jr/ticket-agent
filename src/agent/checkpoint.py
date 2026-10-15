"""The checkpointer: saves the state of every run in Postgres, so a run can pause (waiting for a
human) and continue later, even after the program was closed and started again."""
import os

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

load_dotenv()


def get_checkpointer(url: str | None = None) -> PostgresSaver:
    """Open a Postgres checkpointer (uses DATABASE_URL from .env). Creates its tables if needed.

    The caller owns the connection: close it with `saver.conn.close()` when done."""
    conn = psycopg.connect(url or os.environ["DATABASE_URL"], autocommit=True, prepare_threshold=0,
                           row_factory=dict_row)
    saver = PostgresSaver(conn)
    saver.setup()  # creates the checkpoint tables the first time, does nothing afterwards
    return saver
