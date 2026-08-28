import abc

from email_assistant.gmail import models
from email_assistant.gmail.adapter import GmailServiceAdapter


class BaseGmailEvent(abc.ABC):  # noqa: B024 — marker base class; subclassing is what matters
    """
    A base class for all the events happening in the Gmail Inbox.
    """

    def __init__(self, gmail_service: GmailServiceAdapter):
        self._gmail_service = gmail_service

    def service(self) -> GmailServiceAdapter:
        return self._gmail_service


class MessageAddedEvent(BaseGmailEvent):
    """
    An event that occurs when a new message is added to the Gmail Inbox.
    """

    def __init__(self, gmail_service: GmailServiceAdapter, message: models.Message):
        super().__init__(gmail_service)
        self._message = message

    def message(self) -> models.Message:
        return self._message


class MessageDeletedEvent(BaseGmailEvent):
    """
    An event that occurs when a message is deleted from the Gmail Inbox.
    """

    def __init__(self, gmail_service: GmailServiceAdapter, message_id: str):
        super().__init__(gmail_service)
        self._message_id = message_id
