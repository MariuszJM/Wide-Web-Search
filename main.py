from src.processing import ContentProcessor
from src.utils import save_results, create_output_directory
from config import OUTPUT_FOLDER, PLATFORM, LLM_PROVIDER, LLM_MODEL
from src.search import GoogleSearchEngine, YouTubeSearchEngine
from src.llm import LLMHandler


def get_search_engine(platform):
    if platform == "google":
        return GoogleSearchEngine()
    elif platform == "youtube":
        return YouTubeSearchEngine()
    else:
        raise ValueError("Invalid platform. Choose 'google' or 'youtube'.")


def main():
    search_engine = get_search_engine(PLATFORM)
    llm_handler = LLMHandler(LLM_PROVIDER, LLM_MODEL)
    content_processor = ContentProcessor(llm_handler.llm, llm_handler.llm_json)

    urls = search_engine.fetch_urls()
    source_items = search_engine.load_source_content(urls)

    processed_items = content_processor.process_content(source_items)

    output_dir = create_output_directory(OUTPUT_FOLDER)
    save_results(processed_items, output_dir)


if __name__ == "__main__":
    main()
