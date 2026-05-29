from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os

# Use persistent Render disk path if available, or environment variable, or local fallback
database_dir = "/data"
if os.path.exists(database_dir) and os.access(database_dir, os.W_OK):
    SQLALCHEMY_DATABASE_URL = "sqlite:////data/sheguard.db"
else:
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sheguard.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
