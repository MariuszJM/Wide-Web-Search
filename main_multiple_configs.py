import os
import glob
from src.processing import ContentProcessor
from src.utils import save_results, create_output_directory, load_config
from src.search import get_search_engine
from config import OUTPUT_FOLDER, LLM_PROVIDER, LLM_MODEL, LLM_MAX_TOKENS
from src.llm import LLMHandler


def main(config_file_name):
    input_user = load_config(config_file_name)
    queries = input_user.get("SEARCH_QUERIES")
    max_sources = input_user.get("MAX_SOURCES_PER_SEARCH_QUERY")
    time_horizon = input_user.get("TIME_HORIZON_DAYS")
    content_questions = input_user.get("CONTENT_QUESTIONS")
    max_top_sources = input_user.get("MAX_TOP_SOURCES")
    platform = input_user.get("PLATFORM")

    search_engine = get_search_engine(platform)
    llm_handler = LLMHandler(LLM_PROVIDER, LLM_MODEL)
    content_processor = ContentProcessor(llm_handler, LLM_MAX_TOKENS)

    urls = search_engine.fetch_urls(queries, max_sources, time_horizon)
    source_items = search_engine.load_source_content(urls)

    processed_items = content_processor.process_content(
        source_items, content_questions, max_top_sources
    )

    output_dir = create_output_directory(OUTPUT_FOLDER)
    save_results(processed_items, output_dir)


if __name__ == "__main__":
    config_folder = "user_input_job_offers"
    config_files = glob.glob(os.path.join(config_folder, "*.yaml"))
    config_files.sort()
    for config_file_name in config_files:
        print(f"Processing configuration file: {config_file_name}")
        main(config_file_name)
