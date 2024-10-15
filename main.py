from src.llm import LLMHandler
from src.search import GoogleSearchEngine, YouTubeSearchEngine
from src.processing import ContentProcessor
from src.utils import save_results, create_output_directory
from config import OUTPUT_FOLDER, PLATFORM, LLM_PROVIDER, LLM_MODEL


def main():
    if PLATFORM == "google":
        search_engine = GoogleSearchEngine()
    elif PLATFORM == "youtube":
        search_engine = YouTubeSearchEngine()
    else:
        raise ValueError("Invalid platform. Choose 'google' or 'youtube'.")

    llm_handler = LLMHandler(LLM_PROVIDER, LLM_MODEL)
    content_processor = ContentProcessor(
        llm_handler.llm, llm_handler.llm_json
    ) 

    urls = search_engine.fetch_urls()
    source_items = search_engine.load_source_content(urls)

    processed_items = content_processor.process_content(source_items)

    output_dir = create_output_directory(OUTPUT_FOLDER)
    save_results(processed_items, output_dir)


if __name__ == "__main__":
    main()
