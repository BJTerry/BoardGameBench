from typing import AsyncGenerator, Iterator, List
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic_ai import models, Agent, capture_run_messages
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelMessage
from bgbench.models import Base

# Prevent real API calls during tests
models.ALLOW_MODEL_REQUESTS = False

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
async def test_llm() -> AsyncGenerator[Agent, None]:
    """Fixture providing a TestModel-based LLM for testing"""
    agent = Agent(TestModel())
    yield agent

@pytest.fixture
def capture_messages() -> Iterator[List[ModelMessage]]:
    """Fixture for capturing message flows in tests"""
    with capture_run_messages() as messages:
        yield messages

@pytest.fixture
def mock_db(mocker):
    """Provide mock database session"""
    return mocker.MagicMock()
