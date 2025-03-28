from notion_assistant.api.client import NotionClient
from notion_assistant.memory.processor import LogEntryProcessor
from notion_assistant.memory.manager import MemoryManager
from notion_assistant.memory.insights import InsightGenerator
from notion_assistant.memory.conversation import ConversationManager
import sys
import time
import threading
from typing import Optional, Callable


def rebuild_database():
    """Rebuild the entire database from Notion."""
    client = NotionClient()
    processor = LogEntryProcessor()
    memory_manager = MemoryManager()

    # Clear existing collection
    memory_manager.clear_collection()

    # List shared pages
    print("\nShared Pages:")
    pages = client.list_shared_pages()
    for page in pages:
        print(f"- {page.title} ({page.type})")
        print(f"  URL: {page.url}")
        print(f"  ID: {page.id}\n")

    # Process first page if available
    if pages:
        print("\nProcessing first page...")
        first_page = pages[0]
        page_content = client.get_page_content(first_page.id)

        if page_content:
            # Process page into log entries
            entries = processor.process_page(page_content)
            print(f"\nFound {len(entries)} log entries")

            # Store entries in memory
            for entry in entries:
                entry_id = memory_manager.store_entry(entry)
                print(f"\nStored entry from {entry.date.strftime('%Y-%m-%d')}")
                print(f"Text preview: {entry.raw_text[:200]}...")


def search_database(query: str, top_k: int = 3):
    """Search the existing database."""
    memory_manager = MemoryManager()
    print(f"\nSearching for: '{query}'")
    results = memory_manager.search(query, top_k=top_k)

    print("\nTop results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"ID: {result.entry.id}")
        print(f"Date: {result.entry.date.strftime('%Y-%m-%d')}")
        print(f"Similarity Score: {result.similarity_score:.3f}")
        print(f"Final Score: {result.final_score:.3f}")
        print("Text:")
        print("=" * 80)
        print(result.entry.raw_text)
        print("=" * 80)


def generate_insights(recent_count: int = 20, window_size: int = 7):
    """Generate insights from recent log entries."""
    memory_manager = MemoryManager()
    insight_generator = InsightGenerator()

    # Get all entries from the database
    print("\nGenerating insights...")
    results = memory_manager.search("", top_k=100)  # Get up to 100 entries
    entries = [result.entry for result in results]

    if not entries:
        print("No entries found in the database.")
        return

    # Generate insights
    insights = insight_generator.generate_insights(entries, recent_count, window_size)

    # Save insights
    filename = insight_generator.save_insights(insights)
    print(f"\nInsights saved to: {filename}")

    # Display insights
    print("\nGenerated Insights:")
    for window in insights["windows"]:
        print(
            f"\nDate Range: {window['date_range']['start']} to {window['date_range']['end']}"
        )
        print("\nKey Insights:")
        for insight in window["insights"]:
            print(f"- {insight}")
        print("\nThemes:")
        for theme in window["themes"]:
            print(f"- {theme}")
        print("\nChanges:")
        for change in window["changes"]:
            print(f"- {change}")


def view_latest_insights():
    """View the most recent insights."""
    insight_generator = InsightGenerator()
    insights = insight_generator.load_latest_insights()

    if "error" in insights:
        print(f"\nError: {insights['error']}")
        return

    print("\nLatest Insights:")
    for window in insights["windows"]:
        print(
            f"\nDate Range: {window['date_range']['start']} to {window['date_range']['end']}"
        )
        print("\nKey Insights:")
        for insight in window["insights"]:
            print(f"- {insight}")
        print("\nThemes:")
        for theme in window["themes"]:
            print(f"- {theme}")
        print("\nChanges:")
        for change in window["changes"]:
            print(f"- {change}")


def chat_with_ben():
    """Chat with Ben using the conversation manager."""
    debug_choice = input(
        "\nenable debug mode to see prompts and responses? (y/n, default: n): "
    ).lower()
    debug_mode = debug_choice == "y"

    conversation = ConversationManager(debug=debug_mode)
    print("\nstarting chat with ben. type 'exit' to return to main menu.")

    # Add the welcome message to conversation history so Ben knows it happened
    welcome_msg = (
        "hey there! what would you like to know about your notion logs or projects?"
    )
    conversation.conversation_history.append(
        {"role": "assistant", "content": welcome_msg}
    )

    print(f"\nben: {welcome_msg}")

    while True:
        user_input = input("\nyou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nending chat...")
            break

        try:
            # Process the message without streaming
            print("\nben is thinking...")
            full_response = conversation.chat(user_input, stream_callback=None)

            # Print the complete response
            print(f"\nben: {full_response.lower()}")

            # Extract debug info if present
            if debug_mode and "\n\n[behaviors:" in full_response:
                response_parts = full_response.split("\n\n[behaviors:")
                debug_info = "\n\n[behaviors:" + response_parts[1]
                print(debug_info)

        except Exception as e:
            print(f"\nError: {str(e)}")


def display_thinking_animation(should_continue: Callable[[], bool]):
    """Display a thinking animation while waiting for a response."""
    animation = [".", "..", "..."]
    i = 0
    sys.stdout.write("\nben is thinking")
    sys.stdout.flush()

    while should_continue():
        sys.stdout.write("\rben is thinking" + animation[i % len(animation)])
        sys.stdout.flush()
        i += 1
        time.sleep(0.3)

    # Clear the thinking line when done - make the clear string longer
    sys.stdout.write("\r" + " " * 30 + "\r")  # Increased from 20 to 30 spaces
    sys.stdout.flush()


def main():
    try:
        while True:
            print("\nhi, i'm ben!\n---")
            print("1. rebuild database from notion")
            print("2. search existing database")
            print("3. generate new insights")
            print("4. view latest insights")
            print("5. chat with me")
            print("6. manage memory entries")
            print("7. bye!")

            choice = input("\nenter your choice (1-7): ")

            if choice == "1":
                rebuild_database()
            elif choice == "2":
                query = input("\nenter your search query: ")
                top_k = int(input("how many results do you want? (default 3): ") or "3")
                search_database(query, top_k)
            elif choice == "3":
                recent_count = int(
                    input("\nhow many recent entries to analyze? (default 20): ")
                    or "20"
                )
                window_size = int(
                    input("window size for analysis? (default 7): ") or "7"
                )
                generate_insights(recent_count, window_size)
            elif choice == "4":
                view_latest_insights()
            elif choice == "5":
                chat_with_ben()
            elif choice == "6":
                manage_memory_entries()
            elif choice == "7":
                print("\ngoodbye!")
                break
            else:
                print("\ninvalid choice. please try again.")

    except Exception as e:
        print(f"Error: {e}")


def manage_memory_entries():
    """Manage ChromaDB entries."""
    memory_manager = MemoryManager()

    while True:
        print("\nMemory Entry Management\n---")
        print("1. view all entries")
        print("2. add new entry")
        print("3. delete entry")
        print("4. return to main menu")

        choice = input("\nenter your choice (1-4): ")

        if choice == "1":
            view_all_entries()
        elif choice == "2":
            add_new_entry()
        elif choice == "3":
            delete_entry()
        elif choice == "4":
            break
        else:
            print("\ninvalid choice. please try again.")


def view_all_entries():
    """View all entries in the database."""
    memory_manager = MemoryManager()
    entries = memory_manager.get_all_entries()

    if not entries:
        print("\nNo entries found in the database.")
        return

    print(f"\nFound {len(entries)} entries:")
    for i, entry in enumerate(entries, 1):
        # Limit text preview length
        text_preview = (
            entry.raw_text[:100] + "..."
            if len(entry.raw_text) > 100
            else entry.raw_text
        )
        print(f"\n{i}. Date: {entry.date.strftime('%Y-%m-%d')}")
        print(f"   ID: {entry.id}")
        print(f"   Preview: {text_preview}")


def add_new_entry():
    """Add a new entry to the database."""
    memory_manager = MemoryManager()

    # Get date
    date_input = input("\nEnter date (YYYY-MM-DD, default: today): ")
    if not date_input:
        from datetime import datetime

        date_input = datetime.now().strftime("%Y-%m-%d")

    # Validate date format
    try:
        datetime.strptime(date_input, "%Y-%m-%d")
    except ValueError:
        print("\nInvalid date format. Please use YYYY-MM-DD.")
        return

    # Get text content
    print("\nEnter entry text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if not line and (not lines or not lines[-1]):
            break
        lines.append(line)

    text = "\n".join(lines)

    if not text.strip():
        print("\nEntry text cannot be empty.")
        return

    # Store the entry
    entry_id = memory_manager.add_entry_for_date(date_input, text)

    if entry_id:
        print(f"\nEntry added successfully with ID: {entry_id}")
    else:
        print("\nFailed to add entry.")


def delete_entry():
    """Delete an entry from the database."""
    memory_manager = MemoryManager()

    # First show all entries
    view_all_entries()

    # Get entry ID to delete
    entry_id = input("\nEnter the ID of the entry to delete: ")

    if not entry_id:
        print("\nNo ID provided.")
        return

    # Confirm deletion
    confirm = input(
        f"\nAre you sure you want to delete entry {entry_id}? (y/n): "
    ).lower()

    if confirm != "y":
        print("\nDeletion cancelled.")
        return

    # Delete the entry
    success = memory_manager.delete_entry(entry_id)

    if success:
        print(f"\nEntry {entry_id} deleted successfully.")
    else:
        print(f"\nFailed to delete entry {entry_id}. Entry may not exist.")


if __name__ == "__main__":
    main()
