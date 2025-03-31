import asyncio

from notifypy import Notify
from telegram import Bot
from telegram.error import TelegramError


__all__ = ["send_telegram_message", "send_notification", "send_message"]


async def send_telegram_message(token, chat_id, text=""):
    """
    Send a message to a telegram chat

    Args:
        token: str: Telegram bot token
        chat_id: int or str or list: Chat id or list of chat ids
        text: str: Message to send
    """
    try:
        bot = Bot(token=token)
        if isinstance(chat_id, (int, str)):
            chat_id = [chat_id]
        for id in chat_id:
            await bot.send_message(chat_id=id, text=text)
    except TelegramError as e:
        print(f"Error sending message: {e}")


def send_notification(title, message=""):
    """
    Send a desktop notification

    Args:
        title: str: Title of the notification
        message: str: Message of the notification
    """
    notification = Notify(default_notification_application_name="bbstrading")
    notification.title = title
    notification.message = message
    notification.send()


def send_message(
    title="SIGNAL",
    message="New signal",
    notify_me=False,
    telegram=False,
    token=None,
    chat_id=None,
):
    """
    Send a message to the user

    Args:
        title: str: Title of the message
        message: str: Message of the message
        notify_me: bool: Send a desktop notification
        telegram: bool: Send a telegram message
        token: str: Telegram bot token
        chat_id: int or str or list: Chat id or list of chat ids
    """
    if notify_me:
        send_notification(title, message=message)
    if telegram:
        if token is None or chat_id is None:
            raise ValueError("Token and chat_id must be provided")
        asyncio.run(send_telegram_message(token, chat_id, text=message))

