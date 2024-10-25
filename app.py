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
    default_config = load_config("config.yaml")

    # Display all YAML fields to be edited in Streamlit
    llm_provider = st.sidebar.text_input(
        "LLM Provider", value=default_config.get("llm_provider", "openai")
    )
    llm_model = st.sidebar.text_input(
        "LLM Model", value=default_config.get("llm_model", "gpt-3.5-turbo")
    )
    llm_temperature = st.sidebar.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(default_config.get("llm_temperature", 0.0)),
    )
    llm_max_tokens = st.sidebar.number_input(
        "LLM Max Tokens", min_value=1, value=default_config.get("llm_max_tokens", 1024)
    )

    search_queries = st.sidebar.text_area(
        "Search phrases (one per line)",
        value="\n".join(default_config.get("search_queries", [])),
    ).split("\n")

    specific_questions = st.sidebar.text_area(
        "Specific questions (one per line)",
        value="\n".join(default_config.get("specific_questions", [])),
    ).split("\n")

    platform = st.sidebar.selectbox(
        "Platform",
        options=["google", "youtube"],
        index=["google", "youtube"].index(default_config.get("platform", "google")),
    )

    time_horizon = st.sidebar.number_input(
        "Time horizon (days)",
        min_value=1,
        value=default_config.get("time_horizon_days", 90),
    )

    max_results = st.sidebar.number_input(
        "Maximum number of results",
        min_value=1,
        value=default_config.get("max_results", 5),
    )

    output_folder = st.sidebar.text_input(
        "Output folder", value=default_config.get("output_folder", "results")
    )

    search_config = {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_temperature": llm_temperature,
        "llm_max_tokens": llm_max_tokens,
        "search_queries": search_queries,
        "specific_questions": specific_questions,
        "platform": platform,
        "time_horizon_days": time_horizon,
        "max_results": max_results,
        "output_folder": output_folder,
    }

    config_yaml = yaml.dump(search_config, allow_unicode=True)
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

            # Display configuration used for the search
            st.subheader("Search Configuration")
            st.json(search_config)  # Show the search configuration in JSON format

            # Run the wide search with the updated configuration
            results = run_wide_search(search_config)

            st.success("Wide search completed.")
            st.json(results)

            # Download results option
            zip_bytes = create_zip_file(results, search_config)
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


def run_wide_search(config):
    llm_handler = LLMHandler(
        llm_provider=config["llm_provider"],
        llm_model=config["llm_model"],
        llm_temperature=config["llm_temperature"],
        llm_max_tokens=config["llm_max_tokens"],
    )

    search_engine = get_search_engine(config["platform"])

    urls = search_engine.fetch_urls(
        search_queries=config["search_queries"],
        sources_per_query=config["max_results"],
        time_horizon_days=config["time_horizon_days"],
    )

    source_items = search_engine.load_source_content(urls)

    content_processor = ContentProcessor(llm_handler)
    processed_items = content_processor.process_content(
        source_items,
        specific_questions=config["specific_questions"],
        max_results=config["max_results"],
    )

    output_dir = create_output_directory(config["output_folder"])
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
