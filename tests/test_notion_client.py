"""
Tests for the NotionClient class.
"""

import pytest
from datetime import datetime
from notion_assistant.api.client import NotionClient


def test_list_shared_pages(notion_client):
    """Test listing shared pages."""
    pages = notion_client.list_shared_pages()
    assert isinstance(pages, list)
    if pages:  # If there are any shared pages
        page = pages[0]
        assert hasattr(page, "id")
        assert hasattr(page, "title")
        assert hasattr(page, "type")
        assert hasattr(page, "url")


def test_get_page_content(notion_client):
    """Test getting page content."""
    pages = notion_client.list_shared_pages()
    if pages:
        page_id = pages[0].id
        content = notion_client.get_page_content(page_id)
        assert isinstance(content, dict)
        assert "results" in content
        assert isinstance(content["results"], list)


def test_parse_blocks(notion_client, sample_page_content):
    """Test parsing blocks from page content."""
    blocks = notion_client.parse_blocks(sample_page_content)
    assert isinstance(blocks, list)
    assert len(blocks) > 0

    # Check first block (heading)
    first_block = blocks[0]
    assert first_block.type == "heading_1"
    assert isinstance(first_block.content, str)
    assert first_block.content == "2024-03-28"

    # Check second block (paragraph)
    second_block = blocks[1]
    assert second_block.type == "paragraph"
    assert isinstance(second_block.content, str)
    assert "Sample log entry" in second_block.content
