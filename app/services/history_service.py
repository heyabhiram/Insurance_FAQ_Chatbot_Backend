import sqlite3
from typing import List, Dict
from datetime import datetime
from app.config import settings

class HistoryService:
    """
    Simple SQLite-backed conversation history store for maintaining multi-turn context.
    """

    def __init__(self, db_path=settings.SQLITE_DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_id 
                ON conversation_history (conversation_id);
                """
            )
            conn.commit()

    def add_message(self, conversation_id: str, role: str, content: str):
        """Append a message to a conversation thread."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversation_history (conversation_id, role, content)
                VALUES (?, ?, ?)
                """,
                (conversation_id, role, content),
            )
            conn.commit()

    def get_recent_history(self, conversation_id: str, limit: int = 6) -> List[Dict[str, str]]:
        """
        Fetch recent messages in chronological order for prompt context.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content, created_at
                    FROM conversation_history
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (conversation_id, limit),
            )
            rows = cursor.fetchall()
            return [{"role": row["role"], "content": row["content"]} for row in rows]

    def get_past_user_questions(self, conversation_id: str, limit: int = 20) -> List[str]:
        """
        Fetch previous questions asked specifically by the user in this conversation,
        ordered from most recent to oldest.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT content FROM conversation_history
                WHERE conversation_id = ? AND role = 'user'
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
            rows = cursor.fetchall()
            return [row["content"] for row in rows]

    def clear_history(self, conversation_id: str):
        """Delete all history for a specific conversation ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM conversation_history WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.commit()

history_service = HistoryService()

