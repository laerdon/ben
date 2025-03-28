from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path
from .models import LogEntry
from .llm import OllamaClient


class InsightGenerator:
    def __init__(self, model: str = "llama3.1"):
        self.llm = OllamaClient(model=model)
        self.insights_dir = Path.home() / "notion_assistant_data" / "insights"
        self.insights_dir.mkdir(parents=True, exist_ok=True)

    def _generate_insights_prompt(
        self, entries: List[LogEntry], window_start: int, window_end: int
    ) -> str:
        """Generate a prompt for the LLM to analyze a window of entries."""
        date_range = f"{entries[window_start].date.strftime('%Y-%m-%d')} to {entries[window_end].date.strftime('%Y-%m-%d')}"

        prompt = f"""Analyze these log entries from {date_range} and provide high-level insights about:

1. Changes in interests and focus areas
2. Emerging concepts or themes
3. Shifts in priorities or goals
4. Patterns in decision-making
5. Notable personal or professional developments

Format your response as:
INSIGHTS:
- Key insight 1
- Key insight 2
...

THEMES:
- Theme 1
- Theme 2
...

CHANGES:
- Change 1
- Change 2
...

Log entries:
"""

        # Add entries to prompt
        for entry in entries[window_start : window_end + 1]:
            prompt += f"\n{entry.date.strftime('%Y-%m-%d')}:\n{entry.raw_text}\n"

        return prompt

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM's response into structured insights."""
        insights = {"insights": [], "themes": [], "changes": []}

        current_section = None
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("INSIGHTS:"):
                current_section = "insights"
            elif line.startswith("THEMES:"):
                current_section = "themes"
            elif line.startswith("CHANGES:"):
                current_section = "changes"
            elif line.startswith("-") and current_section:
                insights[current_section].append(line[1:].strip())

        return insights

    def generate_insights(
        self, entries: List[LogEntry], recent_count: int = 20, window_size: int = 7
    ) -> Dict:
        """Generate insights from log entries using sliding windows."""
        if not entries:
            return {"error": "No entries to analyze"}

        # Sort entries by date (newest first)
        sorted_entries = sorted(entries, key=lambda x: x.date, reverse=True)

        # Take the most recent entries
        recent_entries = sorted_entries[:recent_count]

        # Generate insights for each window
        all_insights = []
        for i in range(0, len(recent_entries), window_size):
            window_end = min(i + window_size - 1, len(recent_entries) - 1)
            prompt = self._generate_insights_prompt(recent_entries, i, window_end)

            try:
                response = self.llm._generate(prompt)
                window_insights = self._parse_llm_response(response)
                window_insights["date_range"] = {
                    "start": recent_entries[i].date.isoformat(),
                    "end": recent_entries[window_end].date.isoformat(),
                }
                all_insights.append(window_insights)
            except Exception as e:
                print(f"Error generating insights for window {i}: {e}")

        return {"generated_at": datetime.now().isoformat(), "windows": all_insights}

    def save_insights(self, insights: Dict):
        """Save insights to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.insights_dir / f"insights_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(insights, f, indent=2)

        return filename

    def load_latest_insights(self) -> Dict:
        """Load the most recent insights file."""
        if not self.insights_dir.exists():
            return {"error": "No insights found"}

        # Get the most recent insights file
        insight_files = list(self.insights_dir.glob("insights_*.json"))
        if not insight_files:
            return {"error": "No insights found"}

        latest_file = max(insight_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, "r") as f:
            return json.load(f)
