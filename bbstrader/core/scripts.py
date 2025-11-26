import argparse
import asyncio
import sys
import textwrap
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal

import nltk
from loguru import logger
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from bbstrader.core.data import FinancialNews
from bbstrader.trading.utils import send_telegram_message


def summarize_text(text: str, sentences_count: int = 5) -> str:
    """
    Generate a summary using TextRank algorithm.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)


def format_coindesk_article(article: Dict[str, Any]) -> str:
    if not all(
        k in article
        for k in (
            "body",
            "title",
            "published_on",
            "sentiment",
            "keywords",
            "status",
            "url",
        )
    ):
        return ""
    summary = summarize_text(article["body"], sentences_count=3)
    text = (
        f"📰 {article['title']}\n"
        f"Published Date: {article['published_on']}\n"
        f"Sentiment: {article['sentiment']}\n"
        f"Status: {article['status']}\n"
        f"Keywords: {article['keywords']}\n\n"
        f"🔍 Summary\n"
        f"{textwrap.fill(summary, width=80)}"
        f"\n\n👉 Visit {article['url']} for full article."
    )
    return text


def format_fmp_article(article: Dict[str, Any]) -> str:
    if not all(k in article for k in ("title", "date", "content", "tickers")):
        return ""
    summary = summarize_text(article["content"], sentences_count=3)
    text = (
        f"📰 {article['title']}\n"
        f"Published Date: {article['date']}\n"
        f"Keywords: {article['tickers']}\n\n"
        f"🔍 Summary\n"
        f"{textwrap.fill(summary, width=80)}"
    )
    return text


async def send_articles(
    articles: List[Dict[str, Any]],
    token: str,
    id: str,
    source: Literal["coindesk", "fmp"],
    interval: int = 15,
) -> None:
    for article in articles:
        message = ""
        if source == "coindesk":
            published_on = article.get("published_on")
            if isinstance(
                published_on, datetime
            ) and published_on >= datetime.now() - timedelta(minutes=interval):
                article["published_on"] = published_on.strftime("%Y-%m-%d %H:%M:%S")
                message = format_coindesk_article(article)
        else:
            message = format_fmp_article(article)
        if message == "":
            continue
        await asyncio.sleep(2)  # To avoid hitting Telegram rate limits
        await send_telegram_message(token, id, text=message)


def send_news_feed(unknown: List[str]) -> None:
    HELP_MSG = """
    Send news feed from Coindesk to Telegram channel.
    This script fetches the latest news articles from Coindesk, summarizes them,
    and sends them to a specified Telegram channel at regular intervals.

    Usage:
        python -m bbstrader --run news_feed [options]

    Options:
        -q, --query: The news to look for (default: "")
        -t, --token: Telegram bot token
        -I, --id: Telegram Chat id
            --fmp: Financial Modeling Prop Api Key
        -i, --interval: Interval in minutes to fetch news (default: 15)

    Note:
        The script will run indefinitely, fetching news every 15 minutes.
        Use Ctrl+C to stop the script.
    """

    if "-h" in unknown or "--help" in unknown:
        print(HELP_MSG)
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--query", type=str, default="", help="The news to look for"
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="Telegram bot token",
    )
    parser.add_argument("-I", "--id", type=str, required=True, help="Telegram Chat id")
    parser.add_argument(
        "--fmp", type=str, default="", help="Financial Modeling Prop Api Key"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=15,
        help="Interval in minutes to fetch news (default: 15)",
    )
    args = parser.parse_args(unknown)

    nltk.download("punkt", quiet=True)
    news = FinancialNews()
    fmp_news = news.get_fmp_news(api=args.fmp) if args.fmp else None
    logger.info(f"Starting the News Feed on {args.interval} minutes")
    while True:
        try:
            fmp_articles: List[Dict[str, Any]] = []
            if fmp_news is not None:
                fmp_articles = fmp_news.get_latest_articles(limit=5)
            coindesk_articles = news.get_coindesk_news(query=args.query)
            if coindesk_articles and isinstance(coindesk_articles, list):
                asyncio.run(
                    send_articles(
                        coindesk_articles,  # type: ignore
                        args.token,
                        args.id,
                        "coindesk",
                        interval=args.interval,
                    )
                )
            if len(fmp_articles) != 0:
                asyncio.run(send_articles(fmp_articles, args.token, args.id, "fmp"))
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            logger.info("Stopping the News Feed ...")
            sys.exit(0)
        except Exception as e:
            logger.error(e)
