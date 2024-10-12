from src.llm import initialize_llm
from src.search import search_content
from src.processing import process_content
from src.utils import save_results, create_output_directory
from config import OUTPUT_FOLDER

def main():
    llm, llm_json = initialize_llm()
    source_items = search_content()
    processed_items = process_content(source_items, llm, llm_json)
    output_dir = create_output_directory(OUTPUT_FOLDER)
    save_results(processed_items, output_dir)

if __name__ == "__main__":
    main()
