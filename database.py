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
    def __init__(self, db_path: str = "app.db"):
        """Initialize the database connection and create tables if they don't exist."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the users and questions tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Create users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )
                """)
                # Create questions table with user_id foreign key
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS questions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        user_question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                """)
                conn.commit()
                logger.info("Database initialized successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")

    def get_db_connection(self):
        """Return a database connection with row factory set."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_user(self, name: str, email: str, hashed_password: str) -> bool:
        """Save a new user to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                    (name, email, hashed_password)
                )
                conn.commit()
                logger.info(f"User {email} saved to database.")
                return True
        except sqlite3.IntegrityError:
            logger.error(f"Email {email} already registered.")
            return False
        except sqlite3.Error as e:
            logger.error(f"Error saving user: {e}")
            return False

    def get_user_by_email(self, email: str) -> dict:
        """Retrieve a user by email."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
                user = cursor.fetchone()
                if user:
                    logger.debug(f"User found for email {email}: {list(user.keys())}")
                    return dict(zip(user.keys(), user))
                logger.debug(f"No user found for email {email}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user {email}: {e}")
            return None

    def update_user_password(self, email: str, hashed_password: str) -> bool:
        """Update a user's password."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET password = ? WHERE email = ?",
                    (hashed_password, email)
                )
                conn.commit()
                logger.info(f"Password updated for user {email}.")
                return True
        except sqlite3.Error as e:
            logger.error(f"Error updating password for {email}: {e}")
            return False

    def save_question_answer(self, user_id: int, user_question: str, answer: str):
        """Save a user question and its corresponding answer to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO questions (user_id, user_question, answer) VALUES (?, ?, ?)",
                    (user_id, user_question, answer)
                )
                conn.commit()
                logger.info(f"Question and answer saved for user_id {user_id}.")
        except sqlite3.Error as e:
            logger.error(f"Error saving question and answer: {e}")

    def get_user_questions(self, user_id: int) -> List[Tuple[int, str, str, str]]:
        """Retrieve all question-answer pairs for a specific user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, user_question, answer, timestamp FROM questions WHERE user_id = ?",
                    (user_id,)
                )
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving questions for user_id {user_id}: {e}")
            return []