
from pydantic import BaseModel, Field


class Message(BaseModel):
    subject: str
    date: str
    content: str
    sender: str
    recipients: str
    cc: str | None
    bcc: str | None


class Thread(BaseModel):
    id: str
    history_id: int = Field(
        description="The ID of the last history record that modified this thread."
    )
    messages: list[Message]
