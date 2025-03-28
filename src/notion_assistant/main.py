from notion_assistant.api.client import NotionClient
from notion_assistant.memory.processor import LogEntryProcessor
from notion_assistant.memory.manager import MemoryManager


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
    for result in results:
        print(f"\nDate: {result.entry.date.strftime('%Y-%m-%d')}")
        print(f"Similarity Score: {result.similarity_score:.3f}")
        print(f"Final Score: {result.final_score:.3f}")
        print(f"Text: {result.entry.raw_text[:200]}...")


def main():
    try:
        while True:
            print("\nNotion Assistant")
            print("1. Rebuild database from Notion")
            print("2. Search existing database")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ")

            if choice == "1":
                rebuild_database()
            elif choice == "2":
                query = input("\nEnter your search query: ")
                top_k = int(input("How many results do you want? (default 3): ") or "3")
                search_database(query, top_k)
            elif choice == "3":
                print("\nGoodbye!")
                break
            else:
                print("\nInvalid choice. Please try again.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
