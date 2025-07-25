# src/database.py

"""Database setup and operations for CryptoPulse."""
import sqlite3
import pandas as pd
from datetime import datetime
import os
import logging

class CryptoPulseDB:
    def __init__(self, db_path='db/cryptopulse.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Reddit posts table (unchanged)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reddit_posts (
                    id TEXT PRIMARY KEY,
                    subreddit TEXT,
                    title TEXT,
                    content TEXT,
                    score INTEGER,
                    num_comments INTEGER,
                    created_utc REAL,
                    url TEXT
                )
            """)
            # Twitter posts table (ensure these columns exist if creating)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS twitter_posts (
                    id TEXT PRIMARY KEY,
                    username TEXT,
                    content TEXT,
                    likes INTEGER,
                    retweets INTEGER,
                    replies INTEGER,
                    created_at REAL,
                    url TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def clear_table(self, table_name):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(f'DELETE FROM {table_name}')
            conn.commit()
        finally:
            conn.close()

    def record_exists(self, table_name: str, record_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                f"SELECT 1 FROM {table_name} WHERE id = ? LIMIT 1",
                (record_id,)
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def insert_reddit_post(self, post: dict):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO reddit_posts
                (id, subreddit, title, content, score, num_comments, created_utc, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post['id'],
                post.get('subreddit'),
                post.get('title'),
                post.get('content'),
                post.get('score'),
                post.get('num_comments'),
                post.get('created_utc', datetime.utcnow().timestamp()),
                post.get('url')
            ))
            conn.commit()
        finally:
            conn.close()

    def insert_reddit_posts(self, df: pd.DataFrame) -> int:
        new_count = 0
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            for row in df.itertuples(index=False):
                pid = row.id
                if cursor.execute(
                    "SELECT 1 FROM reddit_posts WHERE id = ? LIMIT 1",
                    (pid,)
                ).fetchone():
                    continue
                ts = (
                    row.created_utc.timestamp()
                    if hasattr(row.created_utc, "timestamp")
                    else row.created_utc
                )
                cursor.execute("""
                    INSERT INTO reddit_posts
                    (id, subreddit, title, content, score, num_comments, created_utc, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pid,
                    getattr(row, 'subreddit', None),
                    getattr(row, 'title', None),
                    getattr(row, 'content', None),
                    getattr(row, 'score', None),
                    getattr(row, 'num_comments', None),
                    ts,
                    getattr(row, 'url', None)
                ))
                new_count += 1
            conn.commit()
        finally:
            conn.close()
        return new_count

    # --- UPDATED TWITTER INSERT METHODS ---
    def insert_twitter_post(self, tweet: dict):
        """Insert a single Tweet into the database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO twitter_posts
                (id, username, content, likes, retweets, replies, created_at, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tweet['id'],
                tweet.get('username'),
                tweet.get('content'),
                tweet.get('likes', 0),
                tweet.get('retweets', 0),
                tweet.get('replies', 0),
                tweet.get('created_at', datetime.utcnow().timestamp()),
                tweet.get('url')
            ))
            conn.commit()
        finally:
            conn.close()

    def insert_twitter_posts(self, df: pd.DataFrame) -> int:
        """
        Batch insert Tweets from a DataFrame.
        Returns the number of newly inserted rows.
        """
        new_count = 0
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            for row in df.itertuples(index=False):
                tid = row.id
                if cursor.execute(
                    "SELECT 1 FROM twitter_posts WHERE id = ? LIMIT 1",
                    (tid,)
                ).fetchone():
                    continue
                ts = (
                    row.created_at.timestamp()
                    if hasattr(row.created_at, "timestamp")
                    else row.created_at
                )
                cursor.execute("""
                    INSERT INTO twitter_posts
                    (id, username, content, likes, retweets, replies, created_at, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tid,
                    getattr(row, 'username', None),
                    getattr(row, 'content', None),
                    getattr(row, 'likes', 0),
                    getattr(row, 'retweets', 0),
                    getattr(row, 'replies', 0),
                    ts,
                    getattr(row, 'url', None)
                ))
                new_count += 1
            conn.commit()
        finally:
            conn.close()
        return new_count


    def insert_news_article(self, article: dict):
        """Insert a single news article into the database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO news_articles
                (id, source, title, content, published_at, url)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                article["id"],
                article.get("source"),
                article.get("title"),
                article.get("content"),
                article.get("published_at", datetime.utcnow()).timestamp() if hasattr(article.get("published_at", datetime.utcnow()), "timestamp") else article.get("published_at"),
                article.get("url")
            ))
            conn.commit()
        finally:
            conn.close()

    def insert_news_articles(self, df: pd.DataFrame) -> int:
        """
        Batch insert news articles from a DataFrame.
        Returns the number of newly inserted rows.
        """
        new_count = 0
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            for row in df.itertuples(index=False):
                aid = row.id
                if cursor.execute(
                    "SELECT 1 FROM news_articles WHERE id = ? LIMIT 1",
                    (aid,)
                ).fetchone():
                    continue
                ts = (
                    row.published_at.timestamp()
                    if hasattr(row.published_at, "timestamp")
                    else row.published_at
                )
                cursor.execute("""
                    INSERT INTO news_articles
                    (id, source, title, content, published_at, url)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    aid,
                    getattr(row, "source", None),
                    getattr(row, "title", None),
                    getattr(row, "content", None),
                    ts,
                    getattr(row, "url", None)
                ))
                new_count += 1
            conn.commit()
        finally:
            conn.close()
        return new_count
