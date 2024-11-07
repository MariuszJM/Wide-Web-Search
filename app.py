import streamlit as st
import yaml
import os
from dotenv import load_dotenv
import logging
from src.processing import ContentProcessor
from src.utils import save_results, create_output_directory, load_config
from src.search import get_search_engine
from src.llm import LLMHandler
import io
import zipfile
from config import OUTPUT_FOLDER, LLM_PROVIDER, LLM_MODEL, LLM_MAX_TOKENS

# Load environment variables
load_dotenv()
LOG_FILE = "app.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


def main():
    st.title("Wide Search Application")
    st.sidebar.header("Search Configuration")

    default_config = load_config("user_input.yaml")

    search_queries = st.sidebar.text_area(
        "SEARCH_QUERIES (one per line)",
        value="\n".join(default_config.get("SEARCH_QUERIES", [])),
    ).split("\n")

    content_questions = st.sidebar.text_area(
        "CONTENT_QUESTIONS (one per line)",
        value="\n".join(default_config.get("CONTENT_QUESTIONS", [])),
    ).split("\n")

    time_horizon = st.sidebar.number_input(
        "TIME_HORIZON_DAYS",
        min_value=1,
        value=default_config.get("TIME_HORIZON_DAYS", 90),
    )

    max_top_sources = st.sidebar.number_input(
        "MAX_TOP_SOURCES",
        min_value=1,
        value=default_config.get("MAX_TOP_SOURCES", 5),
    )

    platform = st.sidebar.selectbox(
        "PLATFORM",
        options=["google", "youtube"],
        index=["google", "youtube"].index(default_config.get("PLATFORM", "google")),
    )

    max_sources_per_search_query = st.sidebar.number_input(
        "MAX_SOURCES_PER_SEARCH_QUERY",
        min_value=1,
        value=default_config.get("MAX_SOURCES_PER_SEARCH_QUERY", 10),
    )

    input_user = {
        "SEARCH_QUERIES": search_queries,
        "CONTENT_QUESTIONS": content_questions,
        "PLATFORM": platform,
        "TIME_HORIZON_DAYS": time_horizon,
        "MAX_TOP_SOURCES": max_top_sources,
        "MAX_SOURCES_PER_SEARCH_QUERY": max_sources_per_search_query,
    }

    config_yaml = yaml.dump(input_user, allow_unicode=True)
    st.sidebar.download_button(
        "Download current configuration as YAML",
        data=config_yaml,
        file_name="current_config.yaml",
        mime="application/x-yaml",
    )

    # Button to start wide search
    if st.button("Run Wide Search"):
        try:
            clear_logs()

            st.info("Starting wide search...")
            logging.info("Wide search started.")

            results = run_wide_search(input_user)

            st.success("Wide search completed.")
            st.json(results)

            # Download results option
            zip_bytes = create_zip_file(results, input_user)
            st.download_button(
                "Download ZIP file",
                data=zip_bytes.getvalue(),
                file_name="wide_search_results.zip",
                mime="application/zip",
            )

            # Clear logs after the run
            clear_logs()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.exception("Error during wide search.")


def run_wide_search(input_user):
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

    return processed_items


def create_zip_file(results, config):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        results_yaml = yaml.dump(results, allow_unicode=True)
        zf.writestr("results.yaml", results_yaml)

        config_yaml = yaml.dump(config, allow_unicode=True)
        zf.writestr("config.yaml", config_yaml)

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as log_file:
                zf.writestr("app.log", log_file.read())

    zip_buffer.seek(0)
    return zip_buffer


def clear_logs():
    """Clear the log file by overwriting it with an empty string."""
    with open(LOG_FILE, "w"):
        pass


if __name__ == "__main__":
    main()
