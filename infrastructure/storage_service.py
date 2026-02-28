import sqlite3
from typing import List, Dict, Any

class StorageService:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    status TEXT DEFAULT 'pending',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    response TEXT,
                    score INTEGER,  -- 1 for Up, -1 for Down
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_message(self, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))

    def get_all_messages(self) -> List[Dict[str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT role, content FROM messages ORDER BY id ASC")
            return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]

    def clear_all(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM tasks")

    def sync_tasks(self, tasks: List[Dict[str, Any]]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tasks")
            for t in tasks:
                conn.execute("INSERT INTO tasks (content, status) VALUES (?, ?)", (t["content"], t.get("status", "pending")))

    def get_tasks(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT content, status FROM tasks")
            return [{"content": row[0], "status": row[1]} for row in cursor.fetchall()]

    def save_feedback(self, query: str, response: str, score: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO feedback (query, response, score) VALUES (?, ?, ?)", (query, response, score))
