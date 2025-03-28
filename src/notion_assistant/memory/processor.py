from typing import List, Optional
from datetime import datetime
import re
from notion_assistant.api.models import NotionBlock, PageContent
from notion_assistant.memory.models import LogEntry


class LogEntryProcessor:
    def __init__(self):
        # Regex patterns for date headings
        self.date_patterns = [
            r"(\d{1,2}/\d{1,2})",  # 3/28
            r"(\d{1,2}-\d{1,2})",  # 3-28
            r"(\d{4}-\d{2}-\d{2})",  # 2024-03-28
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",  # 28 Mar 2024
        ]

        # Month mapping for text dates
        self.month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

    def _is_date_heading(self, block: NotionBlock) -> bool:
        """Check if a block is a date heading."""
        if block.type not in ["heading_1", "heading_2", "heading_3"]:
            return False

        text = "".join(rt.plain_text for rt in block.content.rich_text)
        return any(re.search(pattern, text) for pattern in self.date_patterns)

    def _parse_date(self, text: str) -> Optional[datetime]:
        """Parse date from various formats."""
        # Try each pattern
        for pattern in self.date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)

                # Handle different formats
                if "/" in date_str or "-" in date_str:
                    parts = date_str.replace("/", "-").split("-")
                    if len(parts) == 2:  # 3/28 format
                        month, day = map(int, parts)
                        return datetime(2024, month, day)  # Assuming current year
                    elif len(parts) == 3:  # 2024-03-28 format
                        year, month, day = map(int, parts)
                        return datetime(year, month, day)
                else:  # Text date format (e.g., "28 Mar 2024")
                    parts = date_str.split()
                    if len(parts) == 3:
                        day = int(parts[0])
                        month = self.month_map[parts[1].lower()[:3]]
                        year = int(parts[2])
                        return datetime(year, month, day)
        return None

    def _get_raw_text(self, blocks: List[NotionBlock]) -> str:
        """Convert blocks to raw text."""
        text_parts = []
        for block in blocks:
            if block.content.rich_text:
                text_parts.append(
                    "".join(rt.plain_text for rt in block.content.rich_text)
                )
        return "\n".join(text_parts)

    def process_page(self, page_content: PageContent) -> List[LogEntry]:
        """Process a page's blocks into log entries."""
        entries = []
        current_blocks = []
        current_date = None

        for block in page_content.blocks:
            if self._is_date_heading(block):
                # Save previous entry if exists
                if current_blocks and current_date:
                    entries.append(
                        LogEntry(
                            date=current_date,
                            blocks=current_blocks,
                            raw_text=self._get_raw_text(current_blocks),
                        )
                    )

                # Start new entry
                text = "".join(rt.plain_text for rt in block.content.rich_text)
                current_date = self._parse_date(text)
                current_blocks = [block]
            elif current_date:
                current_blocks.append(block)

        # Add final entry
        if current_blocks and current_date:
            entries.append(
                LogEntry(
                    date=current_date,
                    blocks=current_blocks,
                    raw_text=self._get_raw_text(current_blocks),
                )
            )

        return entries
