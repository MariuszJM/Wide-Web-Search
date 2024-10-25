from abc import ABC, abstractmethod
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_community import GoogleSearchAPIWrapper
import os
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from datetime import datetime, timedelta


class BaseSearchEngine(ABC):
    """Base class for common search engine logic."""

    @abstractmethod
    def fetch_urls(self, queries, max_sources, time_horizon):
        """Method to fetch unique URLs. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def load_documents(self, url):
        """Method to load documents based on the URL. Must be implemented by subclasses."""
        pass

    def load_source_content(self, urls):
        """Method to load the content from a list of URLs using subclass's load_documents."""
        source_items = {}
        for url in urls:
            documents = self.load_documents(url)
            title = documents[0].metadata.get("title", url)
            source_items[title] = {"url": url, "documents": documents, "qa": {}}
        return source_items


class GoogleSearchEngine(BaseSearchEngine):
    """Search engine class for Google."""

    def fetch_urls(self, queries, max_sources, time_horizon):
        search_wrapper = GoogleSearchAPIWrapper()
        unique_urls = set()

        for query in queries:
            results = search_wrapper.results(
                query,
                max_sources,
                search_params={"dateRestrict": f"d{time_horizon}", "gl": "EN"},
            )
            urls = [item["link"] for item in results]
            unique_urls.update(urls)

        return list(unique_urls)

    def load_documents(self, url):
        """Load documents using WebBaseLoader for Google URLs."""
        loader = WebBaseLoader(url)
        return loader.load()


class YouTubeSearchEngine(BaseSearchEngine):
    """Search engine class for YouTube."""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        self.youtube = self.authenticate_youtube()

    def authenticate_youtube(self):
        return build("youtube", "v3", developerKey=self.api_key)

    def fetch_urls(self, queries, max_sources, time_horizon):
        unique_urls = set()

        for query in queries:
            response = (
                self.youtube.search()
                .list(
                    q=query,
                    part="snippet",
                    maxResults=max_sources,
                    type="video",
                    publishedAfter=(
                        datetime.now() - timedelta(days=time_horizon)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
                .execute()
            )

            for item in response["items"]:
                video_id = item["id"]["videoId"]
                url = f"https://www.youtube.com/watch?v={video_id}"
                unique_urls.add(url)

        return list(unique_urls)

    def load_documents(self, url):
        """Load documents by fetching YouTube transcripts."""
        video_id = url.split("watch?v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        content = " ".join([entry["text"] for entry in transcript])
        return [Document(page_content=content, metadata={"title": url, "url": url})]


def get_search_engine(platform):
    if platform == "google":
        return GoogleSearchEngine()
    elif platform == "youtube":
        return YouTubeSearchEngine()
    else:
        raise ValueError("Invalid platform. Choose 'google' or 'youtube'.")
