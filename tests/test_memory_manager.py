"""
Tests for the MemoryManager class.
"""

import pytest
from datetime import datetime
from notion_assistant.memory.manager import MemoryManager
from notion_assistant.memory.models import LogEntry


def test_store_entry(memory_manager):
    """Test storing a log entry in memory."""
    # Create a sample entry
    entry = LogEntry(
        date=datetime(2024, 3, 28),
        summary="Test entry",
        importance=0.8,
        raw_text="Test raw text",
        blocks=[],  # Empty blocks for testing
    )

    # Store the entry
    entry_id = memory_manager.store_entry(entry)
    assert isinstance(entry_id, str)
    assert len(entry_id) > 0


def test_search(memory_manager):
    """Test searching through stored entries."""
    # Store a test entry
    entry = LogEntry(
        date=datetime(2024, 3, 28),
        summary="Test entry about important decisions",
        importance=0.8,
        raw_text="Test raw text about important decisions",
        blocks=[],  # Empty blocks for testing
    )
    memory_manager.store_entry(entry)

    # Test search
    results = memory_manager.search("important decisions", top_k=1)
    assert isinstance(results, list)
    assert len(results) > 0

    result = results[0]
    assert hasattr(result, "entry")
    assert hasattr(result, "final_score")
    assert isinstance(result.final_score, float)
    assert result.final_score > 0


def test_recency_bias(memory_manager):
    """Test that recency bias is applied correctly."""
    # Store entries from different dates
    old_entry = LogEntry(
        date=datetime(2024, 1, 1),
        summary="Old entry",
        importance=0.8,
        raw_text="Old text",
        blocks=[],  # Empty blocks for testing
    )
    new_entry = LogEntry(
        date=datetime(2024, 3, 28),
        summary="New entry",
        importance=0.8,
        raw_text="New text",
        blocks=[],  # Empty blocks for testing
    )

    memory_manager.store_entry(old_entry)
    memory_manager.store_entry(new_entry)

    # Search and verify recency bias
    results = memory_manager.search("entry", top_k=2)
    assert len(results) == 2
    # Newer entry should have a higher final score
    assert results[0].final_score >= results[1].final_score
