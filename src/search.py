from datetime import datetime, timedelta
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_community import GoogleSearchAPIWrapper
from config import SEARCH_QUERIES, SOURCES_PER_QUERY, TIME_HORIZON_DAYS, PLATFORM
import os
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


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
        results = search_wrapper.results(query, SOURCES_PER_QUERY, search_params={'dateRestrict': f'd{TIME_HORIZON_DAYS}', 'gl': 'EN'})
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
    unique_urls = set()
    source_items = {}

    for query in SEARCH_QUERIES:
        loader = YouTubeDocumentLoader(query=query, limit=SOURCES_PER_QUERY, time_horizon=TIME_HORIZON_DAYS)

        try:
            documents = loader.load()
        except Exception as e:
            print(f"Error loading documents for query {query}: {e}")
            continue

        for document in documents:
            url = document.metadata.get('url')
            title = document.metadata.get('title', url)

            if url not in unique_urls:
                unique_urls.add(url)

                publish_date_str = document.metadata.get('publish_date')
                if publish_date_str:
                    publish_date = datetime.strptime(publish_date_str, '%Y-%m-%d %H:%M:%S')
                    if datetime.now() - publish_date > timedelta(days=TIME_HORIZON_DAYS):
                        continue

                source_items[title] = {
                    'url': url,
                    'documents': [document],
                    'qa': {}
                }

    return source_items


class YouTubeDocumentLoader(BaseLoader):
    def __init__(self, query: str, limit: int, time_horizon: int) -> None:
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        self.query = query
        self.limit = limit
        self.time_horizon = time_horizon
        self.youtube = self.authenticate_youtube()

    def authenticate_youtube(self):
        return build("youtube", "v3", developerKey=self.api_key)

    def load(self) -> list[Document]:
        sources = self.fetch_source_items()
        filtered_sources = self.filter_low_quality_sources(sources)[:self.limit]
        return self.collect_source_details(filtered_sources)

    def fetch_source_items(self):
        response = self.youtube.search().list(q=self.query, part="snippet", maxResults=4*self.limit, type="video").execute()
        return response["items"]

    def filter_low_quality_sources(self, sources):
        filtered_sources = []
        for item in sources:
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            published_at = snippet["publishedAt"]
            days_since_creation = self.calculate_days_passed(published_at)
            if days_since_creation <= self.time_horizon:
                url = f"https://www.youtube.com/watch?v={video_id}"
                filtered_sources.append((snippet["title"], url, video_id))
        return filtered_sources

    def calculate_days_passed(self, date: str) -> int:
        created_date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
        return (datetime.now() - created_date).days

    def collect_source_details(self, sources) -> list[Document]:
        data = []
        for title, url, video_id in sources:
            content = self.fetch_detailed_content(video_id)
            data.append(Document(page_content=content, metadata={"title": title, "url": url}))
        return data

    def fetch_detailed_content(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript])
        except Exception:
            return "Transcript not available."
