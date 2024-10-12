from datetime import datetime, timedelta
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from config import SEARCH_QUERIES, SOURCES_PER_QUERY, TIME_HORIZON_DAYS, PLATFORM

def search_content():
    if PLATFORM == 'google':
        return search_google()
    elif PLATFORM == 'youtube':
        return search_youtube()
    else:
        raise ValueError("Invalid platform. Choose 'google' or 'youtube'.")

def search_google():
    search_wrapper = GoogleSearchAPIWrapper()
    unique_urls = set()
    for query in SEARCH_QUERIES:
        results = search_wrapper.results(query, SOURCES_PER_QUERY, search_params={
            'dateRestrict': f'd{TIME_HORIZON_DAYS}',
            'gl': 'EN'
        })
        urls = [item['link'] for item in results]
        unique_urls.update(urls)
    source_items = {}
    for url in unique_urls:
        loader = WebBaseLoader(url)
        documents = loader.load()
        title = documents[0].metadata.get('title', url)
        source_items[title] = {
            'url': url,
            'documents': documents,
            'qa': {}
        }
    return source_items

def search_youtube():
    youtube_tool = YouTubeSearchTool()
    unique_urls = set()
    for query in SEARCH_QUERIES:
        urls_str = youtube_tool.run(query, 4 * SOURCES_PER_QUERY)
        urls = set(eval(urls_str))
        unique_urls.update(urls)
    source_items = {}
    for url in unique_urls:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        try:
            documents = loader.load()
        except Exception as e:
            print(f"Loading {url} failed: {e}")
            continue
        title = documents[0].metadata.get('title', url)
        publish_date_str = documents[0].metadata.get('publish_date')
        if publish_date_str:
            publish_date = datetime.strptime(publish_date_str, '%Y-%m-%d %H:%M:%S')
            if datetime.now() - publish_date > timedelta(days=TIME_HORIZON_DAYS):
                continue
        source_items[title] = {
            'url': url,
            'documents': documents,
            'qa': {}
        }
    return source_items
