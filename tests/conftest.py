from typing import AsyncGenerator
import os
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from bgbench.data.models import Base
from tests.test_llm import TestLLM

# Mark all tests as requiring anyio for async support
pytestmark = pytest.mark.anyio


@pytest.fixture
def db_session():
    """Provide a database session for testing"""
    # Use in-memory SQLite for tests by default
    # Test-specific PostgreSQL connection can be set with TEST_DATABASE_URL env var
    test_db_url = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")

    engine = create_engine(test_db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest_asyncio.fixture
async def test_llm() -> AsyncGenerator[TestLLM, None]:
    """Fixture providing a TestLLM instance for testing"""
    llm = TestLLM()
    yield llm


@pytest.fixture
def mock_db(mocker):
    """Provide mock database session"""
    return mocker.MagicMock()
