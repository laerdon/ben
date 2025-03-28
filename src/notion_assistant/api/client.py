from typing import List, Dict, Any, Optional
from notion_client import Client
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from .models import NotionBlock, BlockContent, RichText, PageContent


class NotionPage(BaseModel):
    id: str
    title: str
    url: str
    type: str


class NotionClient:
    def __init__(self):
        load_dotenv()
        self.token = os.getenv("NOTION_TOKEN")
        if not self.token:
            raise ValueError("NOTION_TOKEN not found in environment variables")
        self.client = Client(auth=self.token)

    def list_shared_pages(self) -> List[NotionPage]:
        """List all pages shared with the integration."""
        try:
            response = self.client.search(
                filter={"property": "object", "value": "page"}
            )

            pages = []
            for page in response.get("results", []):
                title = (
                    page.get("properties", {})
                    .get("title", {})
                    .get("title", [{}])[0]
                    .get("plain_text", "Untitled")
                )
                pages.append(
                    NotionPage(
                        id=page["id"], title=title, url=page.get("url", ""), type="page"
                    )
                )
            return pages
        except Exception as e:
            print(f"Error listing pages: {e}")
            return []

    def list_shared_databases(self) -> List[NotionPage]:
        """List all databases shared with the integration."""
        try:
            response = self.client.search(
                filter={"property": "object", "value": "database"}
            )

            databases = []
            for database in response.get("results", []):
                title = database.get("title", [{}])[0].get("plain_text", "Untitled")
                databases.append(
                    NotionPage(
                        id=database["id"],
                        title=title,
                        url=database.get("url", ""),
                        type="database",
                    )
                )
            return databases
        except Exception as e:
            print(f"Error listing databases: {e}")
            return []

    def _parse_rich_text(self, rich_text_list: List[Dict]) -> List[RichText]:
        """Parse rich text content from Notion blocks."""
        return [
            RichText(
                plain_text=item.get("plain_text", ""),
                annotations=item.get("annotations", {}),
                href=item.get("href"),
            )
            for item in rich_text_list
        ]

    def _parse_block_content(self, block: Dict) -> BlockContent:
        """Parse block content based on block type."""
        content = BlockContent()

        # Handle different block types
        block_type = block.get("type", "")
        block_data = block.get(block_type, {})

        # Common rich text parsing
        if "rich_text" in block_data:
            content.rich_text = self._parse_rich_text(block_data["rich_text"])

        # Handle specific block types
        if block_type == "to_do":
            content.checked = block_data.get("checked", False)
        elif block_type == "bulleted_list_item":
            content.items = [rt.plain_text for rt in content.rich_text]
        elif block_type == "numbered_list_item":
            content.items = [rt.plain_text for rt in content.rich_text]

        return content

    def _get_block_children(self, block_id: str) -> List[NotionBlock]:
        """Recursively get all child blocks for a given block."""
        try:
            blocks = []
            has_more = True
            start_cursor = None

            while has_more:
                # Get blocks with pagination
                response = self.client.blocks.children.list(
                    block_id=block_id, start_cursor=start_cursor
                )

                for block in response.get("results", []):
                    block_type = block.get("type", "")
                    has_children = block.get("has_children", False)

                    parsed_block = NotionBlock(
                        id=block["id"],
                        type=block_type,
                        content=self._parse_block_content(block),
                        has_children=has_children,
                    )

                    if has_children:
                        parsed_block.children = self._get_block_children(block["id"])

                    blocks.append(parsed_block)

                # Check if there are more pages
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")

            return blocks
        except Exception as e:
            print(f"Error getting block children: {e}")
            return []

    def get_page_content(self, page_id: str) -> Optional[PageContent]:
        """Retrieve all content from a specific page."""
        try:
            # Get page metadata
            page = self.client.pages.retrieve(page_id=page_id)

            # Get page title
            title = (
                page.get("properties", {})
                .get("title", {})
                .get("title", [{}])[0]
                .get("plain_text", "Untitled")
            )

            # Get all blocks
            blocks = self._get_block_children(page_id)

            return PageContent(title=title, blocks=blocks)
        except Exception as e:
            print(f"Error getting page content: {e}")
            return None

    def print_page_content(self, page_id: str):
        """Print the content of a page in a readable format."""
        content = self.get_page_content(page_id)
        if not content:
            print("Could not retrieve page content")
            return

        print(f"\nPage: {content.title}\n")

        def print_block(block: NotionBlock, level: int = 0):
            indent = "  " * level

            # Print block content
            if block.content.rich_text:
                text = "".join(rt.plain_text for rt in block.content.rich_text)
                print(f"{indent}{text}")

            # Print child blocks
            if block.children:
                for child in block.children:
                    print_block(child, level + 1)

        for block in content.blocks:
            print_block(block)
