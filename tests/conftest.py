from typing import AsyncGenerator
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from bgbench.models import Base
from tests.test_llm import TestLLM

# Mark all tests as requiring anyio for async support
pytestmark = pytest.mark.anyio

@pytest.fixture
def db_session():
    """Provide a database session for testing"""
    engine = create_engine('sqlite:///:memory:')
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
