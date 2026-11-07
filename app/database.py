import sqlite3
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from flask import current_app


def get_connection():

    database_url = current_app.config.get("DATABASE_URL")

    if database_url:
        return psycopg2.connect(
            database_url,
            cursor_factory=RealDictCursor
        )

    database_path = Path(current_app.config["DATABASE_PATH"])
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    return connection


def init_db(app):

    database_url = app.config.get("DATABASE_URL")

    if database_url:

        connection = psycopg2.connect(database_url)

        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        filename TEXT NOT NULL,
                        predicted_label TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        ai_probability DOUBLE PRECISION NOT NULL,
                        model_name TEXT NOT NULL DEFAULT 'legacy_unrecorded',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cursor.execute("""
                    ALTER TABLE predictions
                    ADD COLUMN IF NOT EXISTS model_name TEXT
                    NOT NULL DEFAULT 'legacy_unrecorded'
                """)

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        return

    database_path = Path(app.config["DATABASE_PATH"])
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                ai_probability REAL NOT NULL,
                model_name TEXT NOT NULL DEFAULT 'legacy_unrecorded',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        ]

        if "model_name" not in columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN model_name TEXT NOT NULL
                DEFAULT 'legacy_unrecorded'
            """)

        connection.commit()


def save_prediction(filename, result):

    database_url = current_app.config.get("DATABASE_URL")
    connection = get_connection()

    try:
        if database_url:

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO predictions (
                        filename, predicted_label, confidence,
                        ai_probability, model_name
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        filename,
                        result["label"],
                        result["confidence"],
                        result["ai_probability"],
                        result["model_name"]
                    )
                )

        else:
            connection.execute(
                """
                INSERT INTO predictions (
                    filename, predicted_label, confidence,
                    ai_probability, model_name
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    result["label"],
                    result["confidence"],
                    result["ai_probability"],
                    result["model_name"]
                )
            )

        connection.commit()

    finally:
        connection.close()


def get_prediction_history():

    database_url = current_app.config.get("DATABASE_URL")
    connection = get_connection()

    try:
        query = """
            SELECT
                id,
                filename,
                predicted_label,
                confidence,
                ai_probability,
                model_name,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 100
        """

        if database_url:
            with connection.cursor() as cursor:
                cursor.execute(query)
                records = cursor.fetchall()

        else:
            records = connection.execute(query).fetchall()

        return records

    finally:
        connection.close()
