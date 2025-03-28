"""
Common test fixtures for the Notion Assistant project.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from notion_assistant.api.client import NotionClient
from notion_assistant.memory.processor import LogEntryProcessor
from notion_assistant.memory.manager import MemoryManager
from notion_assistant.memory.llm import OllamaClient


class MockOllamaClient:
    def analyze_entry(self, text: str, date: str) -> tuple[str, float]:
        return "Test summary", 0.8


class MockChromaCollection:
    def add(self, embeddings, documents, metadatas, ids):
        pass

    def query(self, query_embeddings, n_results):
        return {
            "ids": [["1", "2"]],
            "distances": [[0.8, 0.6]],
            "documents": [["Test text 1", "Test text 2"]],
            "metadatas": [
                [
                    {
                        "date": "2024-03-28T00:00:00",
                        "summary": "Test summary 1",
                        "importance": 0.8,
                    },
                    {
                        "date": "2024-01-01T00:00:00",
                        "summary": "Test summary 2",
                        "importance": 0.6,
                    },
                ]
            ],
        }


class MockChromaClient:
    def get_or_create_collection(self, name, metadata):
        return MockChromaCollection()


class MockSentenceTransformer:
    def encode(self, text):
        return [0.1] * 384  # Return a fixed-size embedding


class MockNotionClient:
    def list_shared_pages(self):
        return [
            MagicMock(
                id="test-page-id",
                title="Test Page",
                type="page",
                url="https://notion.so/test-page",
            )
        ]

    def get_page_content(self, page_id):
        return {
            "results": [
                {
                    "id": "1",
                    "type": "heading_1",
                    "content": {"rich_text": [{"plain_text": "2024-03-28"}]},
                    "has_children": False,
                },
                {
                    "id": "2",
                    "type": "paragraph",
                    "content": {
                        "rich_text": [{"plain_text": "Sample log entry for testing."}]
                    },
                    "has_children": False,
                },
            ]
        }

    def parse_blocks(self, content):
        return [
            MagicMock(type="heading_1", content="2024-03-28"),
            MagicMock(type="paragraph", content="Sample log entry for testing."),
        ]


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM client."""
    return MockOllamaClient()


@pytest.fixture
def notion_client():
    """Fixture providing a NotionClient instance."""
    with patch(
        "notion_assistant.api.client.NotionClient", return_value=MockNotionClient()
    ):
        return NotionClient()


@pytest.fixture
def log_processor(mock_llm):
    """Fixture providing a LogEntryProcessor instance with mock LLM."""
    with patch("notion_assistant.memory.processor.OllamaClient", return_value=mock_llm):
        return LogEntryProcessor(model="mistral")


@pytest.fixture
def memory_manager():
    """Fixture providing a MemoryManager instance."""
    with patch(
        "notion_assistant.memory.manager.chromadb.PersistentClient",
        return_value=MockChromaClient(),
    ), patch(
        "notion_assistant.memory.manager.SentenceTransformer",
        return_value=MockSentenceTransformer(),
    ):
        return MemoryManager()


@pytest.fixture
def sample_page_content():
    """Fixture providing sample page content for testing."""
    return {
        "title": "Test Page",
        "blocks": [
            {
                "id": "1",
                "type": "heading_1",
                "content": {"rich_text": [{"plain_text": "2024-03-28"}]},
                "has_children": False,
            },
            {
                "id": "2",
                "type": "paragraph",
                "content": {
                    "rich_text": [{"plain_text": "Sample log entry for testing."}]
                },
                "has_children": False,
            },
        ],
    }
