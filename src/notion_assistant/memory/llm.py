import requests
from typing import Tuple


class OllamaClient:
    def __init__(
        self, model: str = "llama3.1:latest", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def _generate(self, prompt: str) -> str:
        """Generate text using Ollama."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"]

    def analyze_entry(self, text: str, date: str) -> Tuple[str, float]:
        """Analyze a log entry to generate a summary and importance score."""
        prompt = f"""Analyze this log entry from {date} and provide:
1. A concise summary (max 2 sentences)
2. An importance score between 0 and 1 (where 1 is most important)

Log entry:
{text}

Format your response as:
SUMMARY: <your summary>
IMPORTANCE: <score>

Focus on key events, decisions, and insights. Consider the entry's significance in the context of personal or professional development."""

        response = self._generate(prompt)

        # Parse response
        summary = ""
        importance = 0.5  # default

        for line in response.split("\n"):
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("IMPORTANCE:"):
                try:
                    importance = float(line.replace("IMPORTANCE:", "").strip())
                    importance = max(0.0, min(1.0, importance))  # clamp between 0 and 1
                except ValueError:
                    pass  # keep default if parsing fails

        return summary, importance
