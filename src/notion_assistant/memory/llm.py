import requests
from typing import Tuple, Generator, Optional, Callable


class OllamaClient:
    def __init__(
        self, model: str = "llama3.1", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def _generate(self, prompt: str) -> str:
        """Generate text using Ollama (non-streaming)."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"]

    def _generate_stream(
        self, prompt: str, callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate text using Ollama with streaming.

        Args:
            prompt: The prompt to send to Ollama
            callback: Optional callback function that receives each chunk of text

        Returns:
            The complete generated text
        """
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True},
            stream=True,
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue

            # Parse the JSON from each line
            try:
                chunk_data = requests.utils.json.loads(line)
                if "response" in chunk_data:
                    chunk = chunk_data["response"]
                    full_response += chunk

                    # Call the callback with the new chunk if provided
                    if callback:
                        callback(chunk)
            except Exception as e:
                # Skip any lines that don't parse correctly
                pass

        return full_response

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
