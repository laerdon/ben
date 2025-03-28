# ben

A local assistant that uses the Notion API and a locally hosted LLM (via Ollama) to read and remember information from your Notion pages.

## Features

- Connect to Notion API using integration token
- Read content from shared Notion pages and databases
- (Coming soon) Memory layer for storing context
- (Coming soon) Webhook listener for Notion updates

## Setup

1. Create a Notion integration at https://www.notion.so/my-integrations
2. Copy your integration token
3. Share your Notion pages/databases with the integration
4. Create a `.env` file in the project root with:
   ```
   NOTION_TOKEN=your_integration_token_here
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the initial setup script:

```bash
python src/notion_assistant/main.py
```

## Project Structure

- `src/notion_assistant/`
  - `api/` - Notion API integration
  - `memory/` - Memory layer for storing context
  - `webhooks/` - Webhook listener for Notion updates
- `tests/` - Test files
