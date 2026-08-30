import abc

from email_assistant.gmail import models
from email_assistant.gmail.adapter import GmailServiceAdapter


class BaseGmailEvent(abc.ABC):  # noqa: B024 — marker base class; subclassing is what matters
    """
    Gmail 收件箱中所有事件的基类。
    """

    def __init__(self, gmail_service: GmailServiceAdapter):
        self._gmail_service = gmail_service

    def service(self) -> GmailServiceAdapter:
        return self._gmail_service


class MessageAddedEvent(BaseGmailEvent):
    """
    新邮件到达 Gmail 收件箱时触发的事件。
    """

    def __init__(self, gmail_service: GmailServiceAdapter, message: models.Message):
        super().__init__(gmail_service)
        self._message = message

    def message(self) -> models.Message:
        return self._message


class MessageDeletedEvent(BaseGmailEvent):
    """
    邮件从 Gmail 收件箱中删除时触发的事件。
    """

    def __init__(self, gmail_service: GmailServiceAdapter, message_id: str):
        super().__init__(gmail_service)
        self._message_id = message_id
