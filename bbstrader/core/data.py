import json
import re
import ssl
from datetime import datetime
from typing import List
from urllib.request import urlopen

import certifi
import pandas as pd
import praw
import requests
import tweepy
import yfinance as yf
from bs4 import BeautifulSoup
from financetoolkit import Toolkit

__all__ = ["FmpData", "FmpNews", "FinancialNews"]


def _get_search_query(query: str) -> str:
    if " " in query:
        return query
    try:
        name = yf.Ticker(query).info["shortName"]
        return query + " " + name
    except Exception:
        return query


def _find_news(query: str | List[str], text):
    if isinstance(query, str):
        query = query.split(" ")
    pattern = r"\b(?:" + "|".join(map(re.escape, query)) + r")\b"
    if re.search(pattern, text, re.IGNORECASE):
        return True
    return False


def _filter_news(news: List[str], query: str | List[str]) -> List[str]:
    return [text for text in news if _find_news(query, text)]


class FmpNews(object):
    """
    ``FmpNews`` is responsible for retrieving financial news, press releases, and articles from Financial Modeling Prep (FMP).

    ``FmpNews`` provides methods to fetch the latest stock, crypto, forex, and general financial news,
    as well as financial articles and press releases.
    """

    def __init__(self, api: str):
        """
        Args:
            api (str): The API key for accessing FMP's news data.

        Example:
            fmp_news = FmpNews(api="your_api_key_here")
        """
        if api is None:
            raise ValueError("API key is required For FmpNews")
        self.__api = api

    def _jsonparsed_data(self, url):
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(url, context=context) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)

    def _load_news(self, news_type, symbol=None, **kwargs) -> List[dict]:
        params = {"start": "from", "end": "to", "page": "page", "limit": "limit"}
        base_url = f"https://financialmodelingprep.com/stable/news/{news_type}-latest?apikey={self.__api}"
        if news_type == "articles":
            assert symbol is None, ValueError("symbol not supported for articles")
            base_url = f"https://financialmodelingprep.com/stable/fmp-articles?apikey={self.__api}"
        elif symbol is not None:
            base_url = f"https://financialmodelingprep.com/stable/news/{news_type}?symbols={symbol}&apikey={self.__api}"

        for param, value in params.items():
            if kwargs.get(param) is not None:
                base_url += f"&{value.strip()}={kwargs.get(param)}"

        return self._jsonparsed_data(base_url)

    def get_articles(self, **kwargs) -> List[dict]:
        def html_parser(content):
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n")
            return text.replace("\n", "")

        articles = self._load_news("articles", **kwargs)
        df = pd.DataFrame(articles)
        df = df[["title", "date", "content", "tickers"]]
        df["content"] = df["content"].apply(html_parser)
        return df.to_dict(orient="records")

    def get_releases(self, symbol=None, **kwargs):
        return self._load_news("press-releases", symbol, **kwargs)

    def get_stock_news(self, symbol=None, **kwargs):
        return self._load_news("stock", symbol, **kwargs)

    def get_crypto_news(self, symbol=None, **kwargs):
        return self._load_news("crypto", symbol, **kwargs)

    def get_forex_news(self, symbol=None, **kwargs):
        return self._load_news("forex", symbol, **kwargs)

    def _last_date(self, date):
        return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    def parse_news(self, news: List[dict], symbol=None, **kwargs) -> List[str]:
        start = kwargs.get("start")
        end = kwargs.get("end")
        end_date = self._last_date(end) if end is not None else datetime.now().date()

        def parse_record(record):
            return " ".join(
                [
                    record.pop("symbol", ""),
                    record.pop("title", ""),
                    record.pop("text", ""),
                    record.pop("content", ""),
                    record.pop("tickers", ""),
                ]
            )

        parsed_news = []
        for record in news:
            date = record.get("publishedDate")
            published_date = self._last_date(record.get("date", date)).date()
            start_date = (
                self._last_date(start).date() if start is not None else published_date
            )
            if published_date >= start_date and published_date <= end_date:
                if symbol is not None:
                    if record.get("symbol", "") == symbol or symbol in record.get(
                        "tickers", ""
                    ):
                        parsed_news.append(parse_record(record))
                else:
                    parsed_news.append(parse_record(record))
        return parsed_news

    def get_latest_articles(self, articles=None, save=False, **kwargs) -> List[dict]:
        end = kwargs.get("end")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        end_date = self._last_date(end) if end is not None else self._last_date(now)
        if articles is None:
            try:
                articles = pd.read_csv("latest_fmp_articles.csv")
                articles = articles.to_dict(orient="records")
                if self._last_date(articles[0]["date"]) < end_date:
                    articles = self.get_articles(**kwargs)
                else:
                    return articles
            except FileNotFoundError:
                articles = self.get_articles(**kwargs)

        if save and len(articles) > 0:
            df = pd.DataFrame(articles)
            df.to_csv("latest_fmp_articles.csv", index=False)
        return articles

    def get_news(self, query, source="articles", articles=None, symbol=None, **kwargs):
        """
        Retrieves relevant financial news based on the specified source.

        Args:
            query (str): The search query or keyword for filtering news, may also be a ticker.
            source (str, optional): The news source to retrieve from. Defaults to "articles".
                                    Available options: "articles", "releases", "stock", "crypto", "forex".
            articles (list, optional): List of pre-fetched articles to use when source="articles". Defaults to None.
            symbol (str, optional): The financial asset symbol (e.g., "AAPL" for stocks, "BTC" for crypto). Defaults to None.
            **kwargs (dict):
                Additional arguments required for fetching news data. May include:
                - start (str): The start period for news retrieval (YYY-MM-DD)
                - end (str): The end period for news retrieval (YYY-MM-DD)
                - page (int): The number  of page to load  for each news
                - limit (int): Maximum Responses per API Call

        Returns:
            list[dict]: A list of filtered news articles relevant to the query.
                        Returns an empty list if no relevant news is found.
        """
        query = _get_search_query(query)
        source_methods = {
            "articles": lambda: self.get_latest_articles(articles=articles, save=True),
            "releases": lambda: self.get_releases(symbol=symbol, **kwargs),
            "stock": lambda: self.get_stock_news(symbol=symbol, **kwargs),
            "crypto": lambda: self.get_crypto_news(symbol=symbol, **kwargs),
            "forex": lambda: self.get_forex_news(symbol=symbol, **kwargs),
        }
        news_source = source_methods.get(source, lambda: [])()
        news = self.parse_news(news_source, symbol=symbol)
        return _filter_news(news, query)


class FinancialNews(object):
    """
    The FinancialNews class provides methods to fetch financial news, articles, and discussions
    from various sources such as Yahoo Finance, Google Finance, Reddit, and Twitter.
    It also supports retrieving news using Financial Modeling Prep (FMP).

    """

    def _fetch_news(self, url, query, asset_type, n_news, headline_tag) -> List[str]:
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            response = None

        if response is None or response.status_code != 200:
            return []

        query = _get_search_query(query)
        soup = BeautifulSoup(response.text, "html.parser")

        headlines = [
            h.text.strip()
            for h in soup.find_all(headline_tag)
            if h.text and _find_news(query, h.text)
        ]
        return headlines[:n_news]

    def get_yahoo_finance_news(self, query, asset_type="stock", n_news=10):
        """
        Fetches recent Yahoo Finance news headlines for a given financial asset.

        Args:
            query (str): The asset symbol or name (e.g., "AAPL").
            asset_type (str, optional): The type of asset (e.g., "stock", "etf"). Defaults to "stock".
            n_news (int, optional): The number of news headlines to return. Defaults to 10.

        Returns:
            list[str]: A list of Yahoo Finance news headlines relevant to the query.
        """
        url = (
            f"https://finance.yahoo.com/quote/{query}/news"
            if asset_type in ["stock", "etf"]
            else "https://finance.yahoo.com/news"
        )
        return self._fetch_news(url, query, asset_type, n_news, "h3")

    def get_google_finance_news(self, query, asset_type="stock", n_news=10):
        """
        Fetches recent Google Finance news headlines for a given financial asset.

        Args:
            query (str): The asset symbol or name (e.g., "AAPL").
            asset_type (str, optional): The type of asset (e.g., "stock", "crypto"). Defaults to "stock".
            n_news (int, optional): The number of news headlines to return. Defaults to 10.

        Returns:
            list[str]: A list of Google Finance news headlines relevant to the query.
        """
        search_terms = {
            "stock": f"{query} stock OR {query} shares OR {query} market",
            "etf": f"{query} ETF OR {query} fund OR {query} exchange-traded fund",
            "futures": f"{query} futures OR {query} price OR {query} market",
            "commodity": f"{query} price OR {query} futures OR {query} market",
            "forex": f"{query} forex OR {query} exchange rate OR {query} market",
            "crypto": f"{query} cryptocurrency OR {query} price OR {query} market",
            "bond": f"{query} bond OR {query} yield OR {query} interest rate",
            "index": f"{query} index OR {query} stock market OR {query} performance",
        }
        search_query = search_terms.get(asset_type, query)
        url = f"https://news.google.com/search?q={search_query.replace(' ', '+')}"
        return self._fetch_news(url, query, asset_type, n_news, "a")

    def get_reddit_posts(
        self,
        symbol,
        client_id=None,
        client_secret=None,
        user_agent=None,
        asset_class="stock",
        n_posts=10,
    ) -> List[str]:
        """
        Fetches recent Reddit posts related to a financial asset.

        This method queries relevant subreddits for posts mentioning the specified symbol
        and returns posts based on the selected asset class (e.g., stock, forex, crypto).
        The function uses the PRAW library to interact with Reddit's API.

        Args:
            symbol (str): The financial asset's symbol or name to search for.
            client_id (str, optional): Reddit API client ID for authentication.
            client_secret (str, optional): Reddit API client secret.
            user_agent (str, optional): Reddit API user agent.
            asset_class (str, optional): The type of financial asset. Defaults to "stock".
                - "stock": Searches in stock-related subreddits (e.g., wallstreetbets, stocks).
                - "forex": Searches in forex-related subreddits.
                - "commodities": Searches in commodity-related subreddits (e.g., gold, oil).
                - "etfs": Searches in ETF-related subreddits.
                - "futures": Searches in futures and options trading subreddits.
                - "crypto": Searches in cryptocurrency-related subreddits.
                - If an unrecognized asset class is provided, defaults to stock-related subreddits.
            n_posts (int, optional): The number of posts to return per subreddit. Defaults to 10.

        Returns:
            list[str]: A list of Reddit post contents matching the query.
                    Each entry contains the post title and body.
                    If no posts are found or an error occurs, returns an empty list.

        Raises:
            praw.exceptions.PRAWException: If an error occurs while interacting with Reddit's API.

        Example:
            >>> get_reddit_posts(symbol="AAPL", client_id="your_id", client_secret="your_secret", user_agent="your_agent", asset_class="stock", n_posts=5)
            ["Apple stock is rallying today due to strong earnings.", "Should I buy $AAPL now?", ...]

        Notes:
            - Requires valid Reddit API credentials.
        """

        reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )

        subreddit_mapping = {
            "stock": ["wallstreetbets", "stocks", "investing", "StockMarket"],
            "forex": ["Forex", "ForexTrading", "DayTrading"],
            "commodities": ["Commodities", "Gold", "Silverbugs", "oil"],
            "etfs": ["ETFs", "investing"],
            "futures": ["FuturesTrading", "OptionsTrading", "DayTrading"],
            "crypto": ["CryptoCurrency", "Bitcoin", "ethereum", "altcoin"],
        }
        try:
            subreddits = subreddit_mapping.get(asset_class.lower(), ["stocks"])
        except Exception:
            return []

        posts = []
        for sub in subreddits:
            subreddit = reddit.subreddit(sub)
            query = _get_search_query(symbol)
            all_posts = subreddit.search(query, limit=n_posts)
            for post in all_posts:
                text = post.title + " " + post.selftext
                if _find_news(query, text):
                    posts.append(text)
        return posts

    def get_twitter_posts(
        self,
        query,
        asset_type="stock",
        bearer=None,
        api_key=None,
        api_secret=None,
        access_token=None,
        access_secret=None,
        n_posts=10,
    ) -> List[str]:
        """
        Fetches recent tweets related to a financial asset.

        This method queries Twitter for recent posts mentioning the specified asset
        and filters the results based on the asset type (e.g., stock, forex, crypto).
        The function uses the Tweepy API to fetch tweets and returns a list of tweet texts.

        Args:
            query (str): The main keyword to search for (e.g., a stock ticker or asset name).
            asset_type (str, optional): The type of financial asset. Defaults to "stock".
                - "stock": Searches for tweets mentioning the stock or shares.
                - "forex": Searches for tweets mentioning foreign exchange (forex) or currency.
                - "crypto": Searches for tweets mentioning cryptocurrency or related terms.
                - "commodity": Searches for tweets mentioning commodities or futures trading.
                - "index": Searches for tweets mentioning stock market indices.
                - "bond": Searches for tweets mentioning bonds or fixed income securities.
                - If an unrecognized asset type is provided, defaults to general finance-related tweets.
            bearer (str, optional): Twitter API bearer token for authentication.
            api_key (str, optional): Twitter API consumer key.
            api_secret (str, optional): Twitter API consumer secret.
            access_token (str, optional): Twitter API access token.
            access_secret (str, optional): Twitter API access token secret.
            n_posts (int, optional): The number of tweets to return. Defaults to 10.

        Returns:
            list[str]: A list of up to `n_posts` tweet texts matching the query.
                    If no tweets are found or an API error occurs, returns an empty list.

        Raises:
            tweepy.TweepyException: If an error occurs while making the Twitter API request.

        Example:
            >>> get_twitter_posts(query="AAPL", asset_type="stock", bearer="YOUR_BEARER_TOKEN", n_posts=5)
            ["Apple stock surges after strong earnings!", "Is $AAPL a buy at this price?", ...]
        """
        client = tweepy.Client(
            bearer_token=bearer,
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret,
        )
        asset_queries = {
            "stock": f"{query} stock OR {query} shares -is:retweet lang:en",
            "forex": f"{query} forex OR {query} currency -is:retweet lang:en",
            "crypto": f"{query} cryptocurrency OR {query} crypto OR #{query} -is:retweet lang:en",
            "commodity": f"{query} commodity OR {query} futures OR {query} trading -is:retweet lang:en",
            "index": f"{query} index OR {query} market -is:retweet lang:en",
            "bond": f"{query} bonds OR {query} fixed income -is:retweet lang:en",
        }
        # Get the correct query based on the asset type
        search = asset_queries.get(
            asset_type.lower(), f"{query} finance -is:retweet lang:en"
        )
        try:
            tweets = client.search_recent_tweets(
                query=search, max_results=100, tweet_fields=["text"]
            )
            query = _get_search_query(query)
            news = [tweet.text for tweet in tweets.data] if tweets.data else []
            return _filter_news(news, query)[:n_posts]
        except tweepy.TweepyException:
            return []

    def get_fmp_news(self, api=None) -> FmpNews:
        return FmpNews(api=api)


class FmpData(Toolkit):
    """
    FMPData class for fetching data from Financial Modeling Prep API
    using the Toolkit class from financetoolkit package.

    See `financetoolkit` for more details.

    """

    def __init__(self, api_key: str = "", symbols: str | list = "AAPL"):
        super().__init__(tickers=symbols, api_key=api_key)


class DataBendo: ...
