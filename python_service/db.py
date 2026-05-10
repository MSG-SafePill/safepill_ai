import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as PgConnection


@contextmanager
def get_connection() -> Iterator[PgConnection]:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "safepill"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", ""),
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_pill_names() -> list[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pill_name FROM pills ORDER BY pill_name")
            rows = cur.fetchall()
    return [row[0] for row in rows]
