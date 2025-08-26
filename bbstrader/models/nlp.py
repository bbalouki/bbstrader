import contextlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple

import dash
import en_core_web_sm
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
from bbstrader.core.data import FinancialNews
from dash import dcc, html
from dash.dependencies import Input, Output
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

__all__ = [
    "TopicModeler",
    "SentimentAnalyzer",
    "LEXICON",
    "EQUITY_LEXICON",
    "FOREX_LEXICON",
    "COMMODITIES_LEXICON",
    "CRYPTO_LEXICON",
    "BONDS_LEXICON",
    "FINANCIAL_LEXICON",
]


EQUITY_LEXICON = {
    # Strongly Positive Sentiment
    "bullish": 3.0,
    "rally": 2.5,
    "breakout": 2.5,
    "upgrade": 2.5,
    "beat estimates": 3.2,
    "strong earnings": 3.5,
    "record revenue": 3.5,
    "profit surge": 3.2,
    "buyback": 2.5,
    "dividend increase": 2.5,
    "guidance raised": 3.0,
    "expanding market share": 2.5,
    "exceeded expectations": 3.2,
    "all-time high": 3.0,
    "strong fundamentals": 3.0,
    "robust growth": 3.2,
    "cash flow positive": 3.0,
    "market leader": 2.8,
    "acquisition": 2.0,
    "cost-cutting": 1.5,
    "strong guidance": 3.0,
    "positive outlook": 2.8,
    "EPS growth": 2.5,
    "undervalued": 2.0,
    # Moderately Positive Sentiment
    "merger talks": 1.8,
    "strategic partnership": 2.0,
    "shareholder value": 2.2,
    "restructuring": 1.5,
    "capital appreciation": 2.0,
    "competitive advantage": 2.5,
    "economic expansion": 2.0,
    "strong balance sheet": 2.8,
    # Neutral Sentiment
    "consolidation": 0.5,
    "technical correction": -0.8,
    "volatility": -1.0,
    "profit-taking": -0.5,
    "neutral rating": 0.0,
    "steady growth": 1.0,
    # Moderately Negative Sentiment
    "short interest rising": -2.0,
    "debt restructuring": -1.5,
    "share dilution": -2.0,
    "regulatory scrutiny": -2.5,
    "missed expectations": -3.0,
    "guidance lowered": -3.0,
    "cost overruns": -2.0,
    "flat revenue": -1.5,
    "underperformance": -2.5,
    "profit margin decline": -2.2,
    "competitive pressure": -1.8,
    "legal issues": -2.0,
    # Strongly Negative Sentiment
    "bearish": -3.0,
    "sell-off": -3.5,
    "downgrade": -2.8,
    "weak earnings": -3.5,
    "profit warning": -3.5,
    "default risk": -4.0,
    "bankruptcy filing": -4.2,
    "liquidity crisis": -3.8,
    "cut dividend": -2.8,
    "earnings decline": -3.2,
    "stock crash": -4.0,
    "economic slowdown": -2.8,
    "recession fears": -3.5,
    "high debt levels": -3.0,
    "market downturn": -3.2,
    "losses widen": -3.8,
    "credit downgrade": -3.5,
}
FOREX_LEXICON = {
    # Strongly Positive Sentiment
    "hawkish": 3.0,
    "rate hike": 2.8,
    "tightening policy": 2.8,
    "currency appreciation": 2.8,
    "strong labor market": 2.5,
    "GDP expansion": 2.5,
    "economic boom": 3.2,
    "inflation under control": 2.5,
    "positive trade balance": 2.5,
    "fiscal stimulus": 2.8,
    "interest rate hike": 2.8,
    "capital inflows": 2.5,
    "strong consumer spending": 2.5,
    "foreign investment inflow": 2.5,
    # Moderately Positive Sentiment
    "interest rate decision": 1.5,
    "central bank intervention": 2.2,
    "GDP growth": 2.2,
    "trade surplus": 2.2,
    "moderate inflation": 1.8,
    "foreign capital influx": 2.0,
    "economic stability": 2.0,
    "currency stabilization": 2.0,
    "improving employment": 2.0,
    "positive business confidence": 2.0,
    # Neutral Sentiment
    "monetary policy": 0.0,
    "exchange rate fluctuation": 0.0,
    "interest rate unchanged": 0.0,
    "trade negotiations": 0.5,
    "stable inflation": 0.5,
    # Moderately Negative Sentiment
    "trade deficit": -2.2,
    "currency depreciation": -2.5,
    "inflation risk": -2.0,
    "economic slowdown": -2.5,
    "high fiscal deficit": -2.2,
    "sovereign debt concerns": -2.5,
    "capital outflows": -2.2,
    "weak consumer confidence": -2.0,
    "soft labor market": -2.0,
    "rising unemployment": -2.5,
    # Strongly Negative Sentiment
    "dovish": -3.0,
    "rate cut": -2.8,
    "quantitative easing": -3.2,
    "recession fears": -3.5,
    "market turmoil": -3.5,
    "economic contraction": -3.2,
    "currency crisis": -3.8,
    "sovereign default": -4.0,
    "credit rating downgrade": -3.5,
    "financial instability": -3.5,
    "debt crisis": -3.8,
    "hyperinflation": -4.0,
}
COMMODITIES_LEXICON = {
    # Strongly Positive Sentiment
    "supply shortage": 3.0,
    "OPEC production cut": 3.2,
    "energy crisis": 3.5,
    "oil embargo": 3.8,
    "commodity supercycle": 3.2,
    "gold safe-haven demand": 2.8,
    "inflation hedge": 2.5,
    "weak dollar": 2.8,
    "geopolitical risk": 2.5,
    "strong demand": 3.0,
    "rising crude prices": 3.2,
    "bullish commodity outlook": 3.0,
    "supply constraints": 3.0,
    # Moderately Positive Sentiment
    "rising metal prices": 2.5,
    "higher energy demand": 2.5,
    "limited production capacity": 2.2,
    "low inventory levels": 2.5,
    "export restrictions": 2.0,
    "strategic reserves release": 1.5,
    "drought impact on crops": 2.0,
    "agriculture supply risk": 2.2,
    # Neutral Sentiment
    "market rebalancing": 0.0,
    "seasonal demand": 0.5,
    "commodity price stabilization": 0.5,
    "production levels steady": 0.0,
    # Moderately Negative Sentiment
    "inventory build-up": -2.5,
    "OPEC production increase": -3.0,
    "mining output increase": -2.2,
    "price cap": -2.2,
    "demand destruction": -2.5,
    "falling oil demand": -2.2,
    "oversupply concerns": -2.5,
    "slowing industrial activity": -2.0,
    "crop surplus": -2.0,
    "market correction": -1.5,
    # Strongly Negative Sentiment
    "strong dollar": -2.8,
    "commodity price crash": -3.5,
    "recession-driven demand slump": -3.5,
    "economic downturn impact": -3.5,
    "excess oil production": -3.0,
    "weak commodity prices": -3.0,
    "global trade slowdown": -3.5,
    "deflationary pressure": -3.8,
}
CRYPTO_LEXICON = {
    # Strongly Positive Sentiment
    "bull run": 3.5,
    "institutional adoption": 3.2,
    "mainnet launch": 3.2,
    "layer 2 adoption": 3.0,
    "token burn": 2.8,
    "hash rate increase": 2.8,
    "exchange outflow rising": 2.8,
    "staking rewards increase": 2.5,
    "whale accumulation": 2.5,
    "strong on-chain activity": 2.5,
    "NFT boom": 2.5,
    "defi yield farming": 2.2,
    "crypto ETF approval": 3.5,
    "blockchain upgrade": 3.2,
    "bullish sentiment": 3.0,
    # Moderately Positive Sentiment
    "FOMO": 2.5,
    "airdrops": 2.2,
    "crypto partnerships": 2.2,
    "cross-chain adoption": 2.2,
    "rising transaction volume": 2.0,
    "mass adoption": 2.5,
    "long liquidations": 2.0,
    "staking demand": 2.0,
    "increasing DeFi TVL": 2.5,
    "uptrend confirmation": 2.5,
    # Neutral Sentiment
    "market correction": 0.0,
    "smart contract execution": 0.0,
    "blockchain fork": 0.0,
    "stablecoin issuance": 0.0,
    "on-chain metrics neutral": 0.0,
    "volatility spike": 0.0,
    # Moderately Negative Sentiment
    "exchange inflow rising": -2.5,
    "network congestion": -2.2,
    "liquidity crisis": -2.5,
    "flash crash": -2.5,
    "stablecoin depeg": -2.8,
    "security breach": -2.8,
    "mining ban": -2.5,
    "bearish divergence": -2.5,
    "liquidation cascade": -2.5,
    "funding rates negative": -2.5,
    # Strongly Negative Sentiment
    "bear market": -3.5,
    "whale dumping": -3.2,
    "FUD": -3.0,
    "rug pull": -3.8,
    "smart contract exploit": -3.5,
    "regulatory crackdown": -3.8,
    "exchange insolvency": -4.0,
    "crypto ban": -4.0,
    "market manipulation": -3.5,
    "scam project": -3.8,
    "protocol failure": -3.5,
    "hacked exchange": -3.8,
    "capitulation": -3.5,
}
BONDS_LEXICON = {
    # Strongly Positive Sentiment
    "yields falling": 2.5,
    "credit upgrade": 3.0,
    "investment grade": 3.2,
    "flight to safety": 2.8,
    "bond rally": 2.8,
    "rate cut expectation": 2.8,
    "monetary easing": 2.8,
    "central bank bond purchases": 2.5,
    "bond demand rising": 2.5,
    "strong bond auction": 2.2,
    "stable credit outlook": 2.5,
    # Moderately Positive Sentiment
    "safe-haven demand": 2.2,
    "long-term bond buying": 2.2,
    "falling credit spreads": 2.0,
    "economic slowdown favoring bonds": 2.0,
    "deflationary environment": 2.2,
    "low-rate environment": 2.2,
    # Neutral Sentiment
    "bond market stabilization": 0.0,
    "steady credit ratings": 0.0,
    "balanced bond flows": 0.0,
    "interest rate outlook neutral": 0.0,
    # Moderately Negative Sentiment
    "corporate debt issuance": -2.2,
    "widening credit spreads": -2.2,
    "rate hike expectation": -2.8,
    "rising borrowing costs": -2.5,
    "tightening liquidity": -2.5,
    "bond outflows": -2.2,
    "weaker bond auction": -2.2,
    # Strongly Negative Sentiment
    "yields rising": -3.0,
    "inverted yield curve": -3.2,
    "credit downgrade": -3.5,
    "default risk rising": -3.8,
    "junk bond status": -3.2,
    "inflation concerns": -3.2,
    "liquidity crunch": -3.2,
    "monetary tightening": -3.2,
    "debt ceiling uncertainty": -3.2,
    "sovereign debt crisis": -4.0,
    "bond market crash": -3.8,
    "hyperinflation risk": -3.8,
}
FINANCIAL_LEXICON = {
    **EQUITY_LEXICON,
    **FOREX_LEXICON,
    **COMMODITIES_LEXICON,
    **CRYPTO_LEXICON,
    **BONDS_LEXICON,
}

LEXICON = {
    "stock": EQUITY_LEXICON,
    "etf": EQUITY_LEXICON,
    "future": FINANCIAL_LEXICON,
    "forex": FOREX_LEXICON,
    "crypto": CRYPTO_LEXICON,
    "index": EQUITY_LEXICON,
    "bond": BONDS_LEXICON,
    "commodity": COMMODITIES_LEXICON,
}


class TopicModeler(object):
    def __init__(self):
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

        try:
            self.nlp = en_core_web_sm.load()
            self.nlp.disable_pipes("ner")
        except OSError:
            raise OSError(
                "SpaCy model 'en_core_web_sm' not found. "
                "Please install it using 'python -m spacy download en_core_web_sm'."
            )

    def preprocess_texts(self, texts: list[str]):
        def clean_doc(Doc):
            doc = []
            for t in Doc:
                if not any(
                    [
                        t.is_stop,
                        t.is_digit,
                        not t.is_alpha,
                        t.is_punct,
                        t.is_space,
                        t.lemma_ == "-PRON-",
                    ]
                ):
                    doc.append(t.lemma_)
            return " ".join(doc)

        texts = (text for text in texts)
        clean_texts = []
        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=100, n_process=8), 1):
            clean_texts.append(clean_doc(doc))
        return clean_texts


class SentimentAnalyzer(object):
    """
    A financial sentiment analysis tool that processes and analyzes sentiment
    from news articles, social media posts, and financial reports.

    This class utilizes NLP techniques to preprocess text and apply sentiment
    analysis using VADER (SentimentIntensityAnalyzer) and optional TextBlob
    for enhanced polarity scoring.

    """

    def __init__(self):
        """
        Initializes the SentimentAnalyzer class by downloading necessary
        NLTK resources and loading the SpaCy NLP model.

        - Downloads NLTK tokenization (`punkt`) and stopwords.
        - Loads the `en_core_web_sm` SpaCy model with Named Entity Recognition (NER) disabled.
        - Initializes VADER's SentimentIntensityAnalyzer for sentiment scoring.

        """
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

        self.analyzer = SentimentIntensityAnalyzer()
        self._stopwords = set(stopwords.words("english"))

        try:
            self.nlp = en_core_web_sm.load()
            self.nlp.disable_pipes("ner")
        except OSError:
            raise OSError(
                "SpaCy model 'en_core_web_sm' not found. "
                "Please install it using 'python -m spacy download en_core_web_sm'."
            )
        self.news = FinancialNews()

    def preprocess_text(self, text: str):
        """
        Preprocesses the input text by performing the following steps:
        1. Converts text to lowercase.
        2. Removes URLs.
        3. Removes all non-alphabetic characters (punctuation, numbers, special symbols).
        4. Tokenizes the text into words.
        5. Removes stop words.
        6. Lemmatizes the words using SpaCy, excluding pronouns.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The cleaned and lemmatized text.
        """
        if not isinstance(text, str):
            raise ValueError(
                f"{self.__class__.__name__}: preprocess_text expects a string, got {type(text)}"
            )
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        words = word_tokenize(text)
        words = [word for word in words if word not in self._stopwords]

        doc = self.nlp(" ".join(words))
        words = [t.lemma_ for t in doc if t.lemma_ != "-PRON-"]

        return " ".join(words)

    def analyze_sentiment(self, texts, lexicon=None, textblob=False) -> float:
        """
        Analyzes the sentiment of a list of texts using VADER or TextBlob.

        Steps:
        1. If a custom lexicon is provided, updates the VADER lexicon.
        2. If `textblob` is set to True, computes sentiment using TextBlob.
        3. Otherwise, preprocesses the text and computes sentiment using VADER.
        4. Returns the average sentiment score of all input texts.

        Args:
            texts (list of str): A list of text inputs to analyze.
            lexicon (dict, optional): A custom sentiment lexicon to update VADER's default lexicon.
            textblob (bool, optional): If True, uses TextBlob for sentiment analysis instead of VADER.

        Returns:
            float: The average sentiment score across all input texts.
                   - Positive values indicate positive sentiment.
                   - Negative values indicate negative sentiment.
                   - Zero indicates neutral sentiment.
        """
        if lexicon is not None:
            self.analyzer.lexicon.update(lexicon)
        if textblob:
            blob = TextBlob(" ".join(texts))
            return blob.sentiment.polarity
        sentiment_scores = [
            self.analyzer.polarity_scores(self.preprocess_text(text))["compound"]
            for text in texts
        ]
        avg_sentiment = (
            sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        )
        return avg_sentiment

    def _get_sentiment_for_one_ticker(
        self,
        ticker: str,
        asset_type: str,
        lexicon=None,
        top_news=10,
        **kwargs,
    ) -> float:
        rd_params = {"client_id", "client_secret", "user_agent"}
        fm_params = {"start", "end", "page", "limit"}

        # 1. Collect data from all sources
        yahoo_news = self.news.get_yahoo_finance_news(
            ticker, asset_type=asset_type, n_news=top_news
        )
        google_news = self.news.get_google_finance_news(
            ticker, asset_type=asset_type, n_news=top_news
        )

        reddit_posts = []
        if all(kwargs.get(rd) for rd in rd_params):
            reddit_posts = self.news.get_reddit_posts(
                ticker,
                n_posts=top_news,
                **{k: kwargs.get(k) for k in rd_params},
            )

        coindesk_news = self.news.get_coindesk_news(query=ticker, list_of_str=True)

        fmp_source_news = []
        if kwargs.get("fmp_api"):
            fmp_news_client = self.news.get_fmp_news(kwargs.get("fmp_api"))
            for src in ["articles"]:
                try:
                    source_news = fmp_news_client.get_news(
                        ticker,
                        source=src,
                        symbol=ticker,
                        **{k: kwargs.get(k) for k in fm_params},
                    )
                    fmp_source_news.extend(source_news)
                except Exception:
                    continue

        # 2. Analyze sentiment for each source
        news_sentiment = self.analyze_sentiment(
            yahoo_news + google_news, lexicon=lexicon
        )
        reddit_sentiment = self.analyze_sentiment(
            reddit_posts, lexicon=lexicon, textblob=True
        )
        fmp_sentiment = self.analyze_sentiment(
            fmp_source_news, lexicon=lexicon, textblob=True
        )
        coindesk_sentiment = self.analyze_sentiment(
            coindesk_news, lexicon=lexicon, textblob=True
        )

        # 3. Compute weighted average sentiment score
        sentiments = [
            news_sentiment,
            reddit_sentiment,
            fmp_sentiment,
            coindesk_sentiment,
        ]
        # Count how many sources provided data to get a proper average
        num_sources = sum(
            1
            for source_data in [
                yahoo_news + google_news,
                reddit_posts,
                fmp_source_news,
                coindesk_news,
            ]
            if source_data
        )

        if num_sources == 0:
            return 0.0

        overall_sentiment = sum(sentiments) / num_sources
        return overall_sentiment

    def get_sentiment_for_tickers(
        self,
        tickers: List[str] | List[Tuple[str, str]],
        lexicon=None,
        asset_type="stock",
        top_news=10,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Computes sentiment scores for a list of financial tickers based on news and social media data.

        Process:
        1. Collects news articles and posts related to each ticker from various sources:
           - Yahoo Finance News
           - Google Finance News
           - Reddit posts
           - Financial Modeling Prep (FMP) news
        2. Analyzes sentiment from each source:
           - Uses VADER for Yahoo and Google Finance news.
           - Uses TextBlob for Reddit and FMP news.
        3. Computes an overall sentiment score using a weighted average approach.

        Args:
            tickers (List[str] | List[Tuple[str, str]]): A list of asset tickers to analyze
                - if using tuples, the first element is the ticker and the second is the asset type.
                - if using a single string, the asset type must be specified or the default is "stock".
            lexicon (dict, optional): A custom sentiment lexicon to update VADER's default lexicon.
            asset_type (str, optional): The type of asset, Defaults to "stock",
                supported types include:
                - "stock": Stock symbols (e.g., AAPL, MSFT)
                - "etf": Exchange-traded funds (e.g., SPY, QQQ)
                - "future": Futures contracts (e.g., CL=F for crude oil)
                - "forex": Forex pairs (e.g., EURUSD=X, USDJPY=X)
                - "crypto": Cryptocurrency pairs (e.g., BTC-USD, ETH-USD)
                - "index": Stock market indices (e.g., ^GSPC for S&P 500)
            top_news (int, optional): Number of news articles/posts to fetch per source. Defaults to 10.
            **kwargs: Additional parameters for API authentication and data retrieval, including:
                - fmp_api (str): API key for Financial Modeling Prep.
                - client_id, client_secret, user_agent (str): Credentials for accessing Reddit API.

        Returns:
            Dict[str, float]: A dictionary mapping each ticker to its overall sentiment score.
                             - Positive values indicate positive sentiment.
                             - Negative values indicate negative sentiment.
                             - Zero indicates neutral sentiment.
        Notes:
            The tickers names must follow yahoo finance conventions.
        """

        sentiment_results = {}

        # Suppress stdout/stderr from underlying  libraries during execution
        with open(os.devnull, "w") as devnull:
            with (
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                with ThreadPoolExecutor() as executor:
                    # Map each future to its ticker for easy result lookup
                    future_to_ticker = {}
                    for ticker_info in tickers:
                        # Normalize input to (ticker, asset_type)
                        if isinstance(ticker_info, tuple):
                            ticker_symbol, ticker_asset_type = ticker_info
                        else:
                            ticker_symbol, ticker_asset_type = ticker_info, asset_type

                        if ticker_asset_type not in [
                            "stock",
                            "etf",
                            "future",
                            "forex",
                            "crypto",
                            "index",
                        ]:
                            raise ValueError(
                                f"Unsupported asset type '{ticker_asset_type}' for {ticker_symbol}."
                            )

                        # Submit the job to the thread pool
                        future = executor.submit(
                            self._get_sentiment_for_one_ticker,
                            ticker=ticker_symbol,
                            asset_type=ticker_asset_type,
                            lexicon=lexicon,
                            top_news=top_news,
                            **kwargs,
                        )
                        future_to_ticker[future] = ticker_symbol

                    # Collect results as they are completed
                    for future in as_completed(future_to_ticker):
                        ticker_symbol = future_to_ticker[future]
                        try:
                            sentiment_score = future.result()
                            sentiment_results[ticker_symbol] = sentiment_score
                        except Exception:
                            sentiment_results[ticker_symbol] = (
                                0.0  # Assign a neutral score on error
                            )

        return sentiment_results

    def get_topn_sentiments(self, sentiments, topn=10):
        """
        Retrieves the top and bottom N assets based on sentiment scores.

        Args:
            sentiments (dict): A dictionary mapping asset tickers to their sentiment scores.
            topn (int, optional): The number of top and bottom assets to return. Defaults to 10.

        Returns:
            tuple: A tuple containing two lists:
                - bottom (list of tuples): The `topn` assets with the lowest sentiment scores, sorted in ascending order.
                - top (list of tuples): The `topn` assets with the highest sentiment scores, sorted in descending order.
        """
        sorted_sentiments = sorted(sentiments.items(), key=lambda x: x[1])
        bottom = sorted_sentiments[:topn]
        top = sorted_sentiments[-topn:]
        return bottom, top

    def _sentiment_bar(self, sentiment_dict, top_n=10):
        bottom_stocks, top_stocks = self.get_topn_sentiments(sentiment_dict, topn=top_n)
        top_bottom_stocks = bottom_stocks + top_stocks

        stocks = [x[0] for x in top_bottom_stocks]
        scores = [x[1] for x in top_bottom_stocks]
        colors = ["red" if s < 0 else "green" for s in scores]

        plt.figure(figsize=(12, 6))
        plt.barh(stocks, scores, color=colors)
        plt.axvline(0, color="black", linewidth=1)

        plt.xlabel("Sentiment Score")
        plt.ylabel("Stock Ticker")
        plt.title(f"Top {top_n} Positive & Negative Stock Sentiments")

        plt.show()

    def _sentiment_scatter(self, sentiment_dict):
        df = pd.DataFrame(
            list(sentiment_dict.items()), columns=["Ticker", "Sentiment Score"]
        )
        fig = px.scatter(
            df,
            x=df.index,
            y="Sentiment Score",
            hover_data=["Ticker"],
            color="Sentiment Score",
            color_continuous_scale=["red", "yellow", "green"],
            title="Stock Sentiment Analysis - Interactive Scatter Plot",
        )
        fig.update_layout(xaxis=dict(showticklabels=False))
        fig.show()

    def visualize_sentiments(self, sentiment_dict, mode="bar", top_n=10):
        """
        Visualizes sentiment scores for financial assets using different chart types.

        Visualization Modes:
        - "bar": Displays a bar chart of the top N assets by sentiment score.
        - "scatter": Displays a scatter plot of sentiment scores.

        Args:
            sentiment_dict (dict): A dictionary mapping asset tickers to their sentiment scores.
            mode (str, optional): The type of visualization to generate.
                                  Options: "bar" (default), "scatter".
            top_n (int, optional): The number of top tickers to display in the bar chart.
                                   Only applicable when mode is "bar".

        Returns:
            None: Displays the sentiment visualization.
        """
        if mode == "bar":
            self._sentiment_bar(sentiment_dict, top_n=top_n)
        elif mode == "scatter":
            self._sentiment_scatter(sentiment_dict)

    def display_sentiment_dashboard(
        self,
        tickers,
        asset_type="stock",
        lexicon=None,
        interval=100_000,
        top_n=20,
        **kwargs,
    ):
        """
        Creates and runs a real-time sentiment analysis dashboard for financial assets.

        The dashboard visualizes sentiment scores for given tickers using interactive
        bar and scatter plots. It fetches new sentiment data at specified intervals.

        Args:
            tickers (List[str] | List[Tuple[str, str]]):
                A list of financial asset tickers to analyze.
                - If using tuples, the first element is the ticker and the second is the asset type.
                - If using a single string, the asset type must be specified or defaults to "stock".
            asset_type (str, optional):
                The type of financial asset ("stock", "forex", "crypto"). Defaults to "stock".
            lexicon (dict, optional):
                A custom sentiment lexicon. Defaults to None.
            interval (int, optional):
                The refresh interval (in milliseconds) for sentiment data updates. Defaults to 100000.
            top_n (int, optional):
                The number of top and bottom assets to display in the sentiment bar chart. Defaults to 20.
            **kwargs (dict):
                Additional arguments required for fetching sentiment data. Must include:
                - client_id (str): Reddit API client ID.
                - client_secret (str): Reddit API client secret.
                - user_agent (str): User agent for Reddit API.
                - fmp_api (str): Financial Modeling Prep (FMP) API key.

        Returns:
            None: The function does not return anything but starts a real-time interactive dashboard.

        Example Usage:
            sa = SentimentAnalyzer()
            sa.display_sentiment_dashboard(
                tickers=["AAPL", "TSLA", "GOOGL"],
                asset_type="stock",
                lexicon=my_lexicon,
                display=True,
                interval=5000,
                top_n=10,
                client_id="your_reddit_id",
                client_secret="your_reddit_secret",
                user_agent="your_user_agent",
                fmp_api="your_fmp_api_key",
            )

        Notes:
            - Sentiment analysis is performed using financial news and social media discussions.
            - The dashboard updates in real-time at the specified interval.
            - The dashboard will keep running unless manually stopped (Ctrl+C).
        """

        app = dash.Dash(__name__)

        sentiment_history = {ticker: [] for ticker in tickers}

        # Dash Layout
        app.layout = html.Div(
            children=[
                html.H1("📊 Real-Time Sentiment Dashboard"),
                dcc.Graph(id="top-sentiment-bar"),
                dcc.Graph(id="sentiment-interactive"),
                dcc.Interval(id="interval-component", interval=interval, n_intervals=0),
            ]
        )

        # Update Sentiment Data
        @app.callback(
            [
                Output("top-sentiment-bar", "figure"),
                Output("sentiment-interactive", "figure"),
            ],
            [Input("interval-component", "n_intervals")],
        )
        def update_dashboard(n):
            start_time = time.time()
            sentiment_data = self.get_sentiment_for_tickers(
                tickers,
                lexicon=lexicon,
                asset_type=asset_type,
                top_news=top_n,
                **kwargs,
            )
            elapsed_time = time.time() - start_time
            print(f"Sentiment Fetch Time: {elapsed_time:.2f} seconds")
            timestamp = datetime.now().strftime("%H:%M:%S")
            for stock, score in sentiment_data.items():
                sentiment_history[stock].append(
                    {"timestamp": timestamp, "score": score}
                )
            data = []
            for stock, scores in sentiment_history.items():
                for entry in scores:
                    data.append(
                        {
                            "Ticker": stock,
                            "Time": entry["timestamp"],
                            "Sentiment Score": entry["score"],
                        }
                    )
            df = pd.DataFrame(data)

            # Top Sentiment Bar Chart
            latest_timestamp = df["Time"].max()
            latest_sentiments = (
                df[df["Time"] == latest_timestamp]
                if not df.empty
                else pd.DataFrame(columns=["Ticker", "Sentiment Score"])
            )

            if latest_sentiments.empty:
                bar_chart = px.bar(title="No Sentiment Data Available")
            else:
                # Get top N and bottom N stocks
                bottom_stocks, top_stocks = self.get_topn_sentiments(
                    sentiment_data, topn=top_n
                )
                top_bottom_stocks = bottom_stocks + top_stocks

                stocks = [x[0] for x in top_bottom_stocks]
                scores = [x[1] for x in top_bottom_stocks]

                df_plot = pd.DataFrame({"Ticker": stocks, "Sentiment Score": scores})
                # Horizontal bar chart
                bar_chart = px.bar(
                    df_plot,
                    x="Sentiment Score",
                    y="Ticker",
                    title=f"Top {top_n} Positive & Negative Sentiment Stocks",
                    color="Sentiment Score",
                    color_continuous_scale=["red", "yellow", "green"],
                    orientation="h",
                )
                bar_chart.add_vline(
                    x=0, line_width=2, line_dash="dash", line_color="black"
                )
                bar_chart.update_layout(
                    xaxis_title="Sentiment Score",
                    yaxis_title="Stock Ticker",
                    yaxis=dict(autorange="reversed"),
                    width=1500,
                    height=600,
                )

            # Sentiment Interactive Scatter Plot
            scatter_chart = px.scatter(
                latest_sentiments,
                x=latest_sentiments.index,
                y="Sentiment Score",
                hover_data=["Ticker"],
                color="Sentiment Score",
                color_continuous_scale=["red", "yellow", "green"],
                title="Stock Sentiment Analysis - Interactive Scatter Plot",
            )
            scatter_chart.update_layout(width=1500, height=600)

            return bar_chart, scatter_chart

        app.run()
