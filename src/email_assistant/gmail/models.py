import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Header(BaseModel):
    name: str
    value: str


class MessagePartBody(BaseModel):
    attachment_id: str | None = Field(default=None, alias="attachmentId")
    size: int
    data: str | None = None


class MessagePart(BaseModel):
    part_id: str = Field(alias="partId")
    mime_type: str = Field(alias="mimeType")
    filename: str
    headers: list[Header]
    body: MessagePartBody
    parts: list["MessagePart"] | None = None


class Message(BaseModel):
    id: str
    thread_id: str = Field(alias="threadId")
    label_ids: list[str] | None = Field(default=None, alias="labelIds")
    snippet: str | None = None
    history_id: str | None = Field(default=None, alias="historyId")
    internal_date: str | None = Field(default=None, alias="internalDate")
    payload: MessagePart | None = None
    size_estimate: int | None = Field(default=None, alias="sizeEstimate")
    raw: str | None = None

    def __str__(self):
        return f"Message(id={self.id}, thread_id={self.thread_id}, snippet={self.snippet}, ...)"

    def get_header_value(self, name: str) -> str | None:
        # payload is Optional and headers may be missing on exotic messages
        if self.payload is None or not self.payload.headers:
            return None
        for header in self.payload.headers:
            if header.name.lower() == name.lower():
                return header.value
        return None


class Thread(BaseModel):
    id: str
    snippet: str | None = None
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
