import json
import re
import ssl
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.request import urlopen

import certifi
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from financetoolkit import Toolkit

__all__ = ["FmpData", "FmpNews", "FinancialNews"]


def _require(module_name: str, extra: str):
    """Import an optional dependency on demand with an actionable error."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"'{module_name}' is required for this feature. Install it with: "
            f"pip install 'bbstrader[{extra}]'"
        ) from exc


def _get_search_query(query: str) -> str:
    """Expand a bare ticker into a richer news-search query.

    For a single-word query, appends the instrument's short name (looked up via
    yfinance) to broaden matches; multi-word or empty queries pass through.

    Args:
        query (str): The raw search query or ticker.

    Returns:
        str: The (possibly expanded) search query.
    """
    if " " in query or query == "":
        return query
    try:
        name = yf.Ticker(query).info["shortName"]
        return query + " " + name
    except Exception:
        return query


def _find_news(query: Union[str, List[str]], text: str) -> bool:
    """Return True if ``text`` contains any whole word from ``query``.

    Args:
        query (Union[str, List[str]]): A search term (space-separated) or list
            of terms to match as whole words, case-insensitively.
        text (str): The text to search.

    Returns:
        bool: Whether any query term appears in ``text``.
    """
    if isinstance(query, str):
        query = query.split(" ")
    pattern = r"\b(?:" + "|".join(map(re.escape, query)) + r")\b"
    if re.search(pattern, text, re.IGNORECASE):
        return True
    return False


def _filter_news(news: List[str], query: Union[str, List[str]]) -> List[str]:
    """Return the news items that match ``query``.

    Args:
        news (List[str]): The candidate news texts.
        query (Union[str, List[str]]): The search term(s) to match.

    Returns:
        List[str]: The items containing a query term.
    """
    return [text for text in news if _find_news(query, text)]


class FmpNews(object):
    """
    ``FmpNews`` is responsible for retrieving financial news, press releases, and articles from Financial Modeling Prep (FMP).

    ``FmpNews`` provides methods to fetch the latest stock, crypto, forex, and general financial news,
    as well as financial articles and press releases.
    """

    def __init__(self, api: str) -> None:
        """
        Args:
            api (str): The API key for accessing FMP's news data.

        Example:
            fmp_news = FmpNews(api="your_api_key_here")
        """
        if api is None:
            raise ValueError("API key is required For FmpNews")
        self.__api = api

    def _jsonparsed_data(self, url: str) -> Any:
        """Fetch ``url`` over a verified TLS connection and parse the JSON body.

        Args:
            url (str): The fully-formed FMP endpoint URL including the API key.

        Returns:
            Any: The decoded JSON payload.
        """
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(url, context=context) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)

    def _load_news(
        self, news_type: str, symbol: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Fetch a page of FMP news/articles of a given type.

        Args:
            news_type (str): The FMP feed (for example ``"stock"``, ``"crypto"``,
                ``"forex"``, ``"press-releases"`` or ``"articles"``).
            symbol (Optional[str]): Restrict to one symbol; not allowed for
                ``"articles"``.
            kwargs (Any): Optional ``start``, ``end``, ``page`` and ``limit``
                query parameters.

        Returns:
            List[Dict[str, Any]]: The raw news records.

        Raises:
            ValueError: If ``symbol`` is given for the ``"articles"`` feed.
        """
        params = {"start": "from", "end": "to", "page": "page", "limit": "limit"}
        base_url = f"https://financialmodelingprep.com/stable/news/{news_type}-latest?apikey={self.__api}"
        if news_type == "articles":
            if symbol is not None:
                raise ValueError("symbol not supported for articles")
            base_url = f"https://financialmodelingprep.com/stable/fmp-articles?apikey={self.__api}"
        elif symbol is not None:
            base_url = f"https://financialmodelingprep.com/stable/news/{news_type}?symbols={symbol}&apikey={self.__api}"

        for param, value in params.items():
            if kwargs.get(param) is not None:
                base_url += f"&{value.strip()}={kwargs.get(param)}"

        return self._jsonparsed_data(base_url)

    def get_articles(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Fetch FMP articles with their HTML content stripped to plain text.

        Args:
            kwargs (Any): Optional ``page``/``limit`` paging parameters.

        Returns:
            List[Dict[str, Any]]: Records with ``title``, ``date``, ``content``
            (plain text) and ``tickers``.
        """

        def html_parser(content: str) -> str:
            """Strip HTML markup from ``content`` to a single line of text.

            Args:
                content (str): The raw HTML article body.

            Returns:
                str: The extracted text with newlines removed.
            """
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n")
            return text.replace("\n", "")

        articles = self._load_news("articles", **kwargs)
        df = pd.DataFrame(articles)
        df = df[["title", "date", "content", "tickers"]]
        df["content"] = df["content"].apply(html_parser)
        return df.to_dict(orient="records")  # type: ignore

    def get_releases(
        self, symbol: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Fetch the latest FMP press releases, optionally for one symbol.

        Args:
            symbol (Optional[str]): Restrict to a single symbol.
            kwargs (Any): Optional paging/date parameters (see :meth:`_load_news`).

        Returns:
            List[Dict[str, Any]]: The press-release records.
        """
        return self._load_news("press-releases", symbol, **kwargs)

    def get_stock_news(
        self, symbol: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Fetch the latest FMP stock news, optionally for one symbol.

        Args:
            symbol (Optional[str]): Restrict to a single symbol.
            kwargs (Any): Optional paging/date parameters (see :meth:`_load_news`).

        Returns:
            List[Dict[str, Any]]: The stock-news records.
        """
        return self._load_news("stock", symbol, **kwargs)

    def get_crypto_news(
        self, symbol: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Fetch the latest FMP crypto news, optionally for one symbol.

        Args:
            symbol (Optional[str]): Restrict to a single symbol.
            kwargs (Any): Optional paging/date parameters (see :meth:`_load_news`).

        Returns:
            List[Dict[str, Any]]: The crypto-news records.
        """
        return self._load_news("crypto", symbol, **kwargs)

    def get_forex_news(
        self, symbol: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Fetch the latest FMP forex news, optionally for one symbol.

        Args:
            symbol (Optional[str]): Restrict to a single symbol.
            kwargs (Any): Optional paging/date parameters (see :meth:`_load_news`).

        Returns:
            List[Dict[str, Any]]: The forex-news records.
        """
        return self._load_news("forex", symbol, **kwargs)

    def _last_date(self, date: str) -> datetime:
        """Parse an FMP ``YYYY-MM-DD HH:MM:SS`` timestamp into a datetime.

        Args:
            date (str): The timestamp string to parse.

        Returns:
            datetime: The parsed datetime.
        """
        return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    def parse_news(
        self, news: List[Dict[str, Any]], symbol: Optional[str] = None, **kwargs: Any
    ) -> List[str]:
        """Flatten and date-filter raw news records into text strings.

        Keeps records published within the ``start``/``end`` window and, when a
        ``symbol`` is given, only those mentioning it.

        Args:
            news (List[Dict[str, Any]]): The raw records to parse.
            symbol (Optional[str]): Restrict to records mentioning this symbol.
            kwargs (Any): Optional ``start`` and ``end`` date bounds as
                ``YYYY-MM-DD HH:MM:SS`` strings.

        Returns:
            List[str]: One flattened text string per matching record.
        """
        start = kwargs.get("start")
        end = kwargs.get("end")
        end_date = self._last_date(end) if end is not None else datetime.now().date()

        def parse_record(record: Dict[str, Any]) -> str:
            """Flatten a news record's text fields into a single string.

            Args:
                record (Dict[str, Any]): A raw news record.

            Returns:
                str: The symbol, title, text, content and tickers joined by spaces.
            """
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
            published_date = self._last_date(record.get("date", date)).date()  # type: ignore
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

    def get_latest_articles(
        self,
        articles: Optional[List[Dict[str, Any]]] = None,
        save: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Return the latest FMP articles, using a local CSV cache when fresh.

        Reads ``latest_fmp_articles.csv`` if present and recent enough; otherwise
        downloads fresh articles and, when ``save`` is set, refreshes the cache.

        Args:
            articles (Optional[List[Dict[str, Any]]]): Pre-fetched articles to
                use instead of reading the cache or downloading.
            save (bool): When True, persist the fetched articles to the cache CSV.
            kwargs (Any): Optional ``end`` bound and paging parameters forwarded
                to :meth:`get_articles`.

        Returns:
            List[Dict[str, Any]]: The latest article records.
        """
        end = kwargs.get("end")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        end_date = self._last_date(end) if end is not None else self._last_date(now)
        if articles is None:
            try:
                articles = pd.read_csv("latest_fmp_articles.csv")  # type: ignore
                articles = articles.to_dict(orient="records")  # type: ignore
                if self._last_date(articles[0]["date"]).hour < end_date.hour:  # type: ignore
                    articles = self.get_articles(**kwargs)
                else:
                    return articles  # type: ignore
            except FileNotFoundError:
                articles = self.get_articles(**kwargs)

        if save and len(articles) > 0:
            df = pd.DataFrame(articles)
            df.to_csv("latest_fmp_articles.csv", index=False)
        return articles

    def get_news(
        self,
        query: str,
        source: str = "articles",
        articles: Optional[List[Dict[str, Any]]] = None,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
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
        if symbol is not None:
            symbol = symbol.replace("-", "").split("=")[
                0
            ]  # if symbol is a yahoo finance ticker
        source_methods = {
            "articles": lambda: self.get_latest_articles(
                articles=articles, save=True, **kwargs
            ),
            "releases": lambda: self.get_releases(symbol=symbol, **kwargs),
            "stock": lambda: self.get_stock_news(symbol=symbol, **kwargs),
            "crypto": lambda: self.get_crypto_news(symbol=symbol, **kwargs),
            "forex": lambda: self.get_forex_news(symbol=symbol, **kwargs),
        }
        news_source = source_methods.get(source, lambda: [])()
        if source == "articles":
            symbol = None  # Articles do not require a symbol filter
        news = self.parse_news(news_source, symbol=symbol, **kwargs)
        return _filter_news(news, query)


class FinancialNews(object):
    """
    The FinancialNews class provides methods to fetch financial news, articles, and discussions
    from various sources such as Yahoo Finance, Google Finance, Reddit, Coindesk and Twitter.
    It also supports retrieving news using Financial Modeling Prep (FMP).

    """

    def _fetch_news(
        self, url: str, query: str, n_news: int, headline_tag: str
    ) -> List[str]:
        """Scrape and filter headlines from a news web page.

        Fetches ``url``, extracts headlines from the given HTML tag and returns
        up to ``n_news`` that match the (query-expanded) search terms. Returns an
        empty list on any request failure.

        Args:
            url (str): The page URL to scrape.
            query (str): The search query or ticker used to filter headlines.
            n_news (int): The maximum number of headlines to return.
            headline_tag (str): The HTML tag holding headline text.

        Returns:
            List[str]: The matching headlines, at most ``n_news`` items.
        """
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

    def get_yahoo_finance_news(
        self, query: str, asset_type: str = "stock", n_news: int = 10
    ) -> List[str]:
        """
        Fetches recent Yahoo Finance news headlines for a given financial asset.

        Args:
            query (str): The asset symbol or name (e.g., "AAPL").
            asset_type (str, optional): The type of asset (e.g., "stock", "etf"). Defaults to "stock",
                supported types include:
                - "stock": Stock symbols (e.g., AAPL, MSFT)
                - "etf": Exchange-traded funds (e.g., SPY, QQQ)
                - "future": Futures contracts (e.g., CL=F for crude oil)
                - "forex": Forex pairs (e.g., EURUSD=X, USDJPY=X)
                - "crypto": Cryptocurrency pairs (e.g., BTC-USD, ETH-USD)
                - "index": Stock market indices (e.g., ^GSPC for S&P 500)
            n_news (int, optional): The number of news headlines to return. Defaults to 10.

        Note:
            For commotities and bonds, use the "Future" asset type.

        Returns:
            list[str]: A list of Yahoo Finance news headlines relevant to the query.
        """
        if asset_type == "forex" or asset_type == "future":
            assert "=" in query, (
                "Forex query must contain '=' for currency pairs (e.g., EURUSD=X, CL=F)"
            )
        if asset_type == "crypto":
            assert "-" in query, (
                "Crypto query must contain '-' for crypto pairs (e.g., BTC-USD, ETH-USD)"
            )
        if asset_type == "index":
            assert query.startswith("^"), (
                "Index query must start with '^' (e.g., ^GSPC for S&P 500)"
            )
        url = (
            f"https://finance.yahoo.com/quote/{query}/news"
            if asset_type in ["stock", "etf", "index", "future", "forex"]
            else "https://finance.yahoo.com/news"
        )
        return self._fetch_news(url, query, n_news, "h3")

    def get_google_finance_news(
        self, query: str, asset_type: str = "stock", n_news: int = 10
    ) -> List[str]:
        """
        Fetches recent Google Finance news headlines for a given financial asset.

        Args:
            query (str): The asset symbol or name (e.g., "AAPL").
            asset_type (str, optional): The type of asset (e.g., "stock", "crypto"). Defaults to "stock".
                Supported types include:
                - "stock": Stock symbols (e.g., AAPL, MSFT)
                - "etf": Exchange-traded funds (e.g., SPY, QQQ)
                - "future": Futures contracts (e.g., CL=F or crude oil)
                - "forex": Forex pairs (e.g., EURUSD, USDJPY)
                - "crypto": Cryptocurrency pairs (e.g., BTCUSD, ETHUSD)
            n_news (int, optional): The number of news headlines to return. Defaults to 10.

        Returns:
            list[str]: A list of Google Finance news headlines relevant to the query.
        """
        search_terms = {
            "stock": f"{query} stock OR {query} shares OR {query} market",
            "etf": f"{query} ETF OR {query} fund OR {query} exchange-traded fund",
            "future": f"{query} futures OR {query} price OR {query} market",
            "forex": f"{query} forex OR {query} exchange rate OR {query} market",
            "crypto": f"{query} cryptocurrency OR {query} price OR {query} market",
            "index": f"{query} index OR {query} stock market OR {query} performance",
        }
        search_query = search_terms.get(asset_type, query)
        url = f"https://news.google.com/search?q={search_query.replace(' ', '+')}"
        return self._fetch_news(url, query, n_news, "a")

    def get_reddit_posts(
        self,
        symbol: str,
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
                - "etf": Searches in ETF-related subreddits.
                - "future": Searches in futures and options trading subreddits.
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

        praw = _require("praw", "social")
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_updates=False,
            comment_kind="t1",
            message_kind="t4",
            redditor_kind="t2",
            submission_kind="t3",
            subreddit_kind="t5",
            trophy_kind="t6",
            oauth_url="https://oauth.reddit.com",
            reddit_url="https://www.reddit.com",
            short_url="https://redd.it",
            timeout=16,
            ratelimit_seconds=5,
        )
        assert reddit.read_only
        subreddit_mapping = {
            "stock": ["wallstreetbets", "stocks", "investing", "StockMarket"],
            "forex": ["Forex", "ForexTrading", "DayTrading"],
            "etfs": ["ETFs", "investing"],
            "futures": [
                "FuturesTrading",
                "OptionsTrading",
                "DayTrading",
                "Commodities",
                "Gold",
                "Silverbugs",
                "oil",
            ],
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
        query: str,
        asset_type: str = "stock",
        bearer: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
        n_posts: int = 10,
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
        tweepy = _require("tweepy", "social")
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
            news = [tweet.text for tweet in tweets.data] if tweets.data else []  # type: ignore
            return _filter_news(news, query)[:n_posts]
        except tweepy.TweepyException:
            return []

    def get_fmp_news(self, api: str | None = None) -> FmpNews:
        """Return an :class:`FmpNews` client for the given API key.

        Args:
            api (str | None): The Financial Modeling Prep API key.

        Returns:
            FmpNews: A news client bound to ``api``.
        """
        return FmpNews(api=api)  # type: ignore

    def get_coindesk_news(
        self,
        query="",
        lang: Literal["EN", "ES", "TR", "FR", "JP", "PT"] = "EN",
        limit=10,
        list_of_str=False,
    ) -> List[str] | List[dict]:
        """
        Fetches and filters recent news articles from CoinDesk's News API.

        Args:
            query : str, optional
                A search term to filter articles by title, body, or keywords.
                If empty, all articles are returned without filtering (default is "").

            lang : Literal["EN", "ES", "TR", "FR", "JP", "PT"], optional
                Language in which to fetch news articles. Supported languages:
                English (EN), Spanish (ES), Turkish (TR), French (FR), Japanese (JP), and Portuguese (PT).
                Default is "EN".

            limit : int, optional
                Maximum number of articles to retrieve. Default is 50.

            list_of_str : bool, optional
                If True, returns a list of strings (concatenated article content).
                If False, returns a list of filtered article dictionaries.
                Default is False.

        Returns:
            List[str] | List[dict]
                - If `query` is empty: returns a list of filtered article dictionaries.
                - If `query` is provided:
                    - Returns a list of strings if `list_of_str=True`.
                    - Returns a list of filtered article dictionaries otherwise.

        Each article dictionary contains the following fields:
            - 'published_on': datetime of publication
            - 'title': article headline
            - 'subtitle': secondary headline
            - 'url': direct link to the article
            - 'body': article content
            - 'keywords': associated tags
            - 'sentiment': sentiment label
            - 'status': publication status

        Notes:
            - Articles marked as sponsored are automatically excluded.
        """
        maximum = 100
        if limit > maximum:
            raise ValueError(f"Number of total news articles allowed is {maximum}")
        try:
            response = requests.get(
                "https://data-api.coindesk.com/news/v1/article/list",
                params={"lang": lang, "limit": limit},
                headers={"Content-type": "application/json; charset=UTF-8"},
            )
            response.raise_for_status()
            json_response = response.json()
        except requests.exceptions.RequestException:
            return []
        if (
            response.status_code != 200
            or "Data" not in json_response
            or len(json_response["Data"]) == 0
        ):
            return []
        articles = json_response["Data"]
        to_keep = [
            "PUBLISHED_ON",
            "TITLE",
            "SUBTITLE",
            "URL",
            "BODY",
            "KEYWORDS",
            "SENTIMENT",
            "STATUS",
        ]
        filtered_articles = []
        for article in articles:
            keys = article.keys()
            filtered_articles.append(
                {
                    k.lower(): article[k]
                    if k in keys and k != "PUBLISHED_ON"
                    else datetime.fromtimestamp(article[k])
                    for k in to_keep
                    if article[k] is not None and "sponsored" not in str(article[k])
                }
            )
        if query == "" or len(filtered_articles) == 0:
            return filtered_articles
        to_return = []
        query = _get_search_query(query)
        for article in filtered_articles:
            if not all(k in article for k in ("title", "body", "keywords")):
                continue
            text = article["title"] + " " + article["body"] + " " + article["keywords"]
            if list_of_str and _find_news(query, text=text):
                to_return.append(text)
            if not list_of_str and _find_news(query, text=text):
                to_return.append(article)
        return to_return


class FmpData(Toolkit):
    """
    FMPData class for fetching data from Financial Modeling Prep API
    using the Toolkit class from financetoolkit package.

    See `financetoolkit` for more details.

    """

    def __init__(self, api_key: str = "", symbols: str | list = "AAPL"):
        """Initialise the FMP-backed toolkit for the given symbols.

        Args:
            api_key (str): The Financial Modeling Prep API key.
            symbols (str | list): One symbol or a list of symbols to load.
        """
        super().__init__(tickers=symbols, api_key=api_key)


class DataBendo: ...
