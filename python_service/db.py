import os
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def get_connection() -> Iterator[Any]:
    import psycopg2

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
    from psycopg2.extras import RealDictCursor

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if _table_exists(cur, "pills"):
                cur.execute(
                    """
                    SELECT
                        id,
                        id AS item_id,
                        'MEDICINE' AS item_type,
                        code,
                        pill_name,
                        NULL AS manufacturer,
                        imprint_text
                    FROM pills
                    ORDER BY pill_name
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT
                        id,
                        id AS item_id,
                        'MEDICINE' AS item_type,
                        item_seq AS code,
                        medicine_name AS pill_name,
                        medicine_manufacturer AS manufacturer,
                        concat_ws(
                            ' ',
                            appearance_info ->> 'printFront',
                            appearance_info ->> 'printBack'
                        ) AS imprint_text
                    FROM medicine_master
                    ORDER BY medicine_name
                    """
                )
            rows = cur.fetchall()
    return [dict(row) for row in rows]


def fetch_medication_catalog() -> list[dict[str, str | int | None]]:
    from psycopg2.extras import RealDictCursor

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            rows: list[dict[str, object]] = []
            if _table_exists(cur, "medicine_master"):
                cur.execute(
                    """
                    SELECT
                        id AS item_id,
                        'MEDICINE' AS item_type,
                        item_seq AS code,
                        medicine_name AS item_name,
                        medicine_manufacturer AS manufacturer,
                        concat_ws(
                            ' ',
                            appearance_info ->> 'printFront',
                            appearance_info ->> 'printBack'
                        ) AS searchable_text
                    FROM medicine_master
                    ORDER BY medicine_name
                    """
                )
                rows.extend(cur.fetchall())
            elif _table_exists(cur, "pills"):
                cur.execute(
                    """
                    SELECT
                        id AS item_id,
                        'MEDICINE' AS item_type,
                        code,
                        pill_name AS item_name,
                        manufacturer,
                        imprint_text AS searchable_text
                    FROM pills
                    ORDER BY pill_name
                    """
                )
                rows.extend(cur.fetchall())

            if _table_exists(cur, "supplement_master"):
                cur.execute(
                    """
                    SELECT
                        id AS item_id,
                        'SUPPLEMENT' AS item_type,
                        item_seq AS code,
                        supplement_name AS item_name,
                        supplement_manufacturer AS manufacturer,
                        supplement_name AS searchable_text
                    FROM supplement_master
                    ORDER BY supplement_name
                    """
                )
                rows.extend(cur.fetchall())

    return [dict(row) for row in rows]


def _table_exists(cur: Any, table_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %s
        )
        """,
        (table_name,),
    )
    row = cur.fetchone()
    return bool(row and row["exists"])


def fetch_pill_context(pill_names: list[str]) -> list[dict[str, object]]:
    if not pill_names:
        return []

    with get_connection() as conn:
        from psycopg2.extras import RealDictCursor

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if _table_exists(cur, "pills"):
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
            else:
                return fetch_main_schema_context(cur, pill_names)
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


def fetch_main_schema_context(cur: Any, pill_names: list[str]) -> list[dict[str, object]]:
    cur.execute(
        """
        SELECT id, medicine_name AS pill_name, medicine_manufacturer AS manufacturer, efficacy, precautions
        FROM medicine_master
        WHERE medicine_name = ANY(%s)
        ORDER BY medicine_name
        """,
        (pill_names,),
    )
    medicines = [dict(row) for row in cur.fetchall()]

    cur.execute(
        """
        SELECT id, supplement_name AS pill_name, supplement_manufacturer AS manufacturer, efficacy, precautions
        FROM supplement_master
        WHERE supplement_name = ANY(%s)
        ORDER BY supplement_name
        """,
        (pill_names,),
    )
    supplements = [dict(row) for row in cur.fetchall()]

    result: list[dict[str, object]] = []
    for item in medicines:
        result.append(
            {
                "pillName": item["pill_name"],
                "manufacturer": item["manufacturer"],
                "dosageForm": None,
                "ingredients": [],
                "warnings": [{"type": "precautions", "text": item.get("precautions")}] if item.get("precautions") else [],
                "interactions": [],
                "efficacy": item.get("efficacy"),
            }
        )
    for item in supplements:
        result.append(
            {
                "pillName": item["pill_name"],
                "manufacturer": item["manufacturer"],
                "dosageForm": None,
                "ingredients": [],
                "warnings": [{"type": "precautions", "text": item.get("precautions")}] if item.get("precautions") else [],
                "interactions": [],
                "efficacy": item.get("efficacy"),
            }
        )
    return result
