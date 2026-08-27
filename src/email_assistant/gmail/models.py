import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Header(BaseModel):
    name: str
    value: str


class MessagePartBody(BaseModel):
    attachment_id: Optional[str] = Field(default=None, alias="attachmentId")
    size: int
    data: Optional[str] = None


class MessagePart(BaseModel):
    part_id: str = Field(alias="partId")
    mime_type: str = Field(alias="mimeType")
    filename: str
    headers: list[Header]
    body: MessagePartBody
    parts: Optional[list["MessagePart"]] = None


class Message(BaseModel):
    id: str
    thread_id: str = Field(alias="threadId")
    label_ids: Optional[list[str]] = Field(default=None, alias="labelIds")
    snippet: Optional[str] = None
    history_id: Optional[str] = Field(default=None, alias="historyId")
    internal_date: Optional[str] = Field(default=None, alias="internalDate")
    payload: Optional[MessagePart] = None
    size_estimate: Optional[int] = Field(default=None, alias="sizeEstimate")
    raw: Optional[str] = None

    def __str__(self):
        return f"Message(id={self.id}, thread_id={self.thread_id}, snippet={self.snippet}, ...)"

    def get_header_value(self, name: str) -> Optional[str]:
        for header in self.payload.headers:
            if header.name.lower() == name.lower():
                return header.value
        return None


class Thread(BaseModel):
    id: str
    snippet: Optional[str] = None
    history_id: str = Field(alias="historyId")
    messages: list[Message]


class MessageAdded(BaseModel):
    message: Message


class MessageDeleted(BaseModel):
    message: Message


class History(BaseModel):
    id: str
    messages: list[Message] = Field(default_factory=list)  # noqa
    messages_added: list[MessageAdded] = Field(  # noqa
        alias="messagesAdded", default_factory=list
    )
    messages_deleted: list[MessageDeleted] = Field(  # noqa
        alias="messagesDeleted", default_factory=list
    )


class DecodedMessage(BaseModel):
    message: Message
    content: str

    def __str__(self):
        return f"DecodedMessage(message={self.message}, content={self.content})"
