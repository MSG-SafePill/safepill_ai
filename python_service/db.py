import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.extras import RealDictCursor
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


def fetch_pill_catalog() -> list[dict[str, str | int | None]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, code, pill_name, imprint_text
                FROM pills
                ORDER BY pill_name
                """
            )
            rows = cur.fetchall()
    return [dict(row) for row in rows]


def fetch_pill_context(pill_names: list[str]) -> list[dict[str, object]]:
    if not pill_names:
        return []

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.id, p.pill_name, p.manufacturer, p.dosage_form
                FROM pills p
                WHERE p.pill_name = ANY(%s)
                ORDER BY p.pill_name
                """,
                (pill_names,),
            )
            pills = [dict(row) for row in cur.fetchall()]
            if not pills:
                return []

            pill_ids = [pill["id"] for pill in pills]

            cur.execute(
                """
                SELECT pi.pill_id, i.name, pi.strength_text
                FROM pill_ingredients pi
                JOIN ingredients i ON i.id = pi.ingredient_id
                WHERE pi.pill_id = ANY(%s)
                ORDER BY pi.pill_id, i.name
                """,
                (pill_ids,),
            )
            ingredients_rows = [dict(row) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT w.pill_id, w.warning_type, w.warning_text
                FROM warnings w
                WHERE w.pill_id = ANY(%s)
                ORDER BY w.pill_id, w.id
                """,
                (pill_ids,),
            )
            warnings_rows = [dict(row) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT i.pill_id, i.target_ingredient, i.severity, i.interaction_text
                FROM interactions i
                WHERE i.pill_id = ANY(%s)
                ORDER BY i.pill_id, i.id
                """,
                (pill_ids,),
            )
            interactions_rows = [dict(row) for row in cur.fetchall()]

    ingredients_by_pill: dict[int, list[dict[str, str | None]]] = {}
    for row in ingredients_rows:
        pill_id = int(row["pill_id"])
        ingredients_by_pill.setdefault(pill_id, []).append(
            {"name": row["name"], "strengthText": row["strength_text"]}
        )

    warnings_by_pill: dict[int, list[dict[str, str]]] = {}
    for row in warnings_rows:
        pill_id = int(row["pill_id"])
        warnings_by_pill.setdefault(pill_id, []).append(
            {"type": row["warning_type"], "text": row["warning_text"]}
        )

    interactions_by_pill: dict[int, list[dict[str, str]]] = {}
    for row in interactions_rows:
        pill_id = int(row["pill_id"])
        interactions_by_pill.setdefault(pill_id, []).append(
            {
                "targetIngredient": row["target_ingredient"],
                "severity": row["severity"],
                "text": row["interaction_text"],
            }
        )

    result: list[dict[str, object]] = []
    for pill in pills:
        pill_id = int(pill["id"])
        result.append(
            {
                "pillName": pill["pill_name"],
                "manufacturer": pill["manufacturer"],
                "dosageForm": pill["dosage_form"],
                "ingredients": ingredients_by_pill.get(pill_id, []),
                "warnings": warnings_by_pill.get(pill_id, []),
                "interactions": interactions_by_pill.get(pill_id, []),
            }
        )
    return result
