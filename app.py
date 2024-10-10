import streamlit as st
import yaml
import logging
import os
from dotenv import load_dotenv
from src.utils import create_zip_file, clear_logs
from src.wide_search import run_wide_search

load_dotenv()

LOG_FILE = 'app.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    st.title("Wide Search Application")

    st.sidebar.header("Search Configuration")

    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            default_config = yaml.safe_load(f)

    search_queries = st.sidebar.text_area("Search phrases (one per line)", value='\n'.join(default_config.get('search_queries', []))).split('\n')
    specific_questions = st.sidebar.text_area("Specific questions (one per line)", value='\n'.join(default_config.get('specific_questions', []))).split('\n')
    platform = st.sidebar.selectbox("Platform", options=['google', 'youtube'], index=default_config.get('platform', ['google']).index('google'))
    time_horizon = st.sidebar.number_input("Time horizon (days)", min_value=1, value=default_config.get('time_horizon', 90))
    max_outputs = st.sidebar.number_input("Maximum number of results", min_value=1, value=default_config.get('max_outputs', 5))

    search_config = {
        'Search Queries': search_queries,
        'Specific Questions': specific_questions,
        'Platform': platform,
        'Time Horizon': time_horizon,
        'Max Outputs': max_outputs
    }

    config_yaml = yaml.dump(search_config, allow_unicode=True)
    st.sidebar.download_button("Download current configuration as YAML", data=config_yaml, file_name="current_config.yaml", mime="application/x-yaml")

    if st.button("Run Wide Search"):
        try:
            clear_logs()

            st.info("Starting wide search...")
            logging.info("Wide search started.")
            
            st.subheader("Search Configuration")
            st.json(search_config)
            logging.info(f"Search configuration: {search_config}")
            
            results = run_wide_search(search_queries, specific_questions, platform, time_horizon, max_outputs)
            
            st.success("Wide search completed.")
            st.json(results)

            zip_bytes = create_zip_file(results, search_config)
            st.download_button("Download ZIP file", data=zip_bytes.getvalue(), file_name="wide_search_results.zip", mime="application/zip")

            clear_logs()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.exception("Error during wide search.")

if __name__ == "__main__":
    main()
