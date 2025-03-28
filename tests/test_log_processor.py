"""
Tests for the LogEntryProcessor class.
"""

import pytest
from datetime import datetime
from notion_assistant.memory.processor import LogEntryProcessor
from notion_assistant.api.models import NotionBlock, BlockContent, RichText


def test_process_page(log_processor, sample_page_content):
    """Test processing a page into log entries."""
    entries = log_processor.process_page(sample_page_content)
    assert isinstance(entries, list)
    assert len(entries) > 0

    # Check first entry
    entry = entries[0]
    assert hasattr(entry, "date")
    assert isinstance(entry.date, datetime)
    assert entry.date.strftime("%Y-%m-%d") == "2024-03-28"
    assert hasattr(entry, "summary")
    assert isinstance(entry.summary, str)
    assert hasattr(entry, "importance")
    assert isinstance(entry.importance, float)
    assert 0 <= entry.importance <= 1
    assert hasattr(entry, "raw_text")
    assert isinstance(entry.raw_text, str)


def test_chunk_by_date(log_processor, sample_page_content):
    """Test chunking content by date headings."""
    chunks = log_processor.chunk_by_date(sample_page_content)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

    chunk = chunks[0]
    assert isinstance(chunk, dict)
    assert "date" in chunk
    assert isinstance(chunk["date"], datetime)
    assert chunk["date"].strftime("%Y-%m-%d") == "2024-03-28"
    assert "content" in chunk
    assert isinstance(chunk["content"], list)
    assert len(chunk["content"]) > 0


def test_summarize_chunk(log_processor):
    """Test summarizing a chunk of content."""
    # Create a proper NotionBlock for testing
    block = NotionBlock(
        id="test",
        type="paragraph",
        content=BlockContent(
            rich_text=[RichText(plain_text="Sample log entry for testing.")]
        ),
        has_children=False,
    )

    chunk = {"date": datetime(2024, 3, 28), "content": [block]}

    entry = log_processor.summarize_chunk(chunk)
    assert hasattr(entry, "date")
    assert entry.date == chunk["date"]
    assert hasattr(entry, "summary")
    assert isinstance(entry.summary, str)
    assert len(entry.summary) > 0
    assert hasattr(entry, "importance")
    assert isinstance(entry.importance, float)
    assert 0 <= entry.importance <= 1
