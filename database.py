import sqlite3
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "poker_coach.db"):
        """Initialize the database connection and create the table if it doesn't exist."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the questions table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS questions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info("Database initialized successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")

    def save_question_answer(self, user_question: str, answer: str):
        """Save a user question and its corresponding answer to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO questions (user_question, answer) VALUES (?, ?)",
                    (user_question, answer)
                )
                conn.commit()
                logger.info("Question and answer saved to database.")
        except sqlite3.Error as e:
            logger.error(f"Error saving question and answer: {e}")

    def get_all_questions(self) -> List[Tuple[int, str, str, str]]:
        """Retrieve all question-answer pairs from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, user_question, answer, timestamp FROM questions")
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving questions: {e}")
            return []