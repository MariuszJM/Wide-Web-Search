# app.py
import streamlit as st
import yaml
import os
from dotenv import load_dotenv
import logging
import zipfile
import io

# Loading environment variables
load_dotenv()

# Path to the log file
LOG_FILE = 'app.log'

# Configure logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    
    st.title("Wide Search Application")

    # Sidebar configuration
    st.sidebar.header("Search Configuration")

    # Loading default configuration if it exists
    default_config = {}
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            default_config = yaml.safe_load(f)

    # User input
    search_queries = st.sidebar.text_area("Search phrases (one per line)", value='\n'.join(default_config.get('search_queries', []))).split('\n')
    specific_questions = st.sidebar.text_area("Specific questions (one per line)", value='\n'.join(default_config.get('specific_questions', []))).split('\n')
    platform = st.sidebar.selectbox("Platform", options=['google', 'youtube'], index=default_config.get('platform', ['google']).index('google'))
    time_horizon = st.sidebar.number_input("Time horizon (days)", min_value=1, value=default_config.get('time_horizon', 90))
    max_outputs = st.sidebar.number_input("Maximum number of results", min_value=1, value=default_config.get('max_outputs', 5))

    search_config = {
                'Search Queries': search_queries,
                'Specific Questions': specific_questions,
                'Platform': platform,
                'Time Horizon (days)': time_horizon,
                'Max Outputs': max_outputs
            }

    config_yaml = yaml.dump(search_config, allow_unicode=True)
    st.sidebar.download_button("Download current configuration as YAML", data=config_yaml, file_name="current_config.yaml", mime="application/x-yaml")

    # Button to start wide search
    if st.button("Run Wide Search"):
        try:
            clear_logs()

            st.info("Starting wide search...")
            logging.info("Wide search started.")

            # Display configuration used for the search
            
            st.subheader("Search Configuration")
            st.json(search_config)  # Show the search configuration in JSON format

            # Log the search configuration for debugging or future reference
            logging.info(f"Search configuration: {search_config}")
            
            # Call to the wide search function (returns example result)
            results = run_wide_search(search_queries, specific_questions, platform, time_horizon, max_outputs)
            
            st.success("Wide search completed.")
            st.json(results)  # Display example result in JSON format

            # Download results option
            zip_bytes = create_zip_file(results, search_config)
            st.download_button("Download ZIP file", data=zip_bytes.getvalue(), file_name="wide_search_results.zip", mime="application/zip")

            # Clear logs after the run
            clear_logs()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.exception("Error during wide search.")

def run_wide_search(search_queries, specific_questions, platform, time_horizon, max_outputs):
    # Example result dictionary
    example_result = {
        "title_1": {
            "url": "https://example.com/link_1",
            "content": '''Side content or YouTube video transcript 
                          or detailed summary if the content was big for LLMs input''',
            "summary": "This is a summary of the content",
            "Q&A": {
                "question_1": "answer_1",
                "question_2": "answer_2"
            }
        },
        "title_2": {
            "url": "https://example.com/link_2",
            "content": "Another example of detailed content summary",
            "summary": "This is another summary of different content",
            "Q&A": {
                "question_1": "answer_1 for title_2",
                "question_2": "answer_2 for title_2"
            }
        }
    }
    
    return example_result

def create_zip_file(results, config):
    # Creating a ZIP archive in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
        # Add test result file
        results_yaml = yaml.dump(results, allow_unicode=True)
        zf.writestr('results.yaml', results_yaml)

        # Add configuration file
        config_yaml = yaml.dump(config, allow_unicode=True)
        zf.writestr('config.yaml', config_yaml)

        # Add logs
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as log_file:
                zf.writestr('app.log', log_file.read())

    zip_buffer.seek(0)
    return zip_buffer

def clear_logs():
    """Clear the log file by overwriting it with an empty string."""
    with open(LOG_FILE, 'w'):
        pass  # Open in write mode to overwrite the contents with an empty file

if __name__ == "__main__":
    main()
