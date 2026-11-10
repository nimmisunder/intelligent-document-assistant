
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "documents.db"


def get_connection():

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    return connection


def init_db():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (

                id TEXT PRIMARY KEY,

                original_filename TEXT NOT NULL,

                stored_filename TEXT NOT NULL,

                file_size INTEGER NOT NULL,

                total_pages INTEGER NOT NULL,

                uploaded_at TEXT NOT NULL

            )
        """)

