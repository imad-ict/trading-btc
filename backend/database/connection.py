"""
Database Connection Management.

Provides async PostgreSQL connections via SQLAlchemy 2.0.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from config import get_settings
from .models import Base

logger = logging.getLogger(__name__)

# Global engine instance
_engine = None
AsyncSessionFactory = None


def get_engine():
    """Get or create the async engine."""
    global _engine
    
    if _engine is None:
        settings = get_settings()
        
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            poolclass=NullPool,  # Disable pooling for better async behavior
        )
        logger.info(f"Database engine created for: {settings.database_url.split('@')[-1]}")
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global AsyncSessionFactory
    
    if AsyncSessionFactory is None:
        engine = get_engine()
        AsyncSessionFactory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    
    return AsyncSessionFactory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    
    Usage:
        async with get_db_session() as session:
            result = await session.execute(query)
    """
    factory = get_session_factory()
    session = factory()
    
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_database() -> None:
    """
    Initialize database tables.
    
    Creates all tables defined in models.py if they don't exist.
    """
    engine = get_engine()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables initialized")


async def dispose_engine() -> None:
    """Dispose of the engine connection pool."""
    global _engine, AsyncSessionFactory
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        AsyncSessionFactory = None
        logger.info("Database engine disposed")
