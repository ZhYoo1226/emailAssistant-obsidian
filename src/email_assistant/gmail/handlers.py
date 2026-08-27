import abc
import logging
from typing import Optional

from markdownify import markdownify

from email_assistant.gmail import events
from email_assistant import models
from email_assistant.crew import AutoResponderCrew

logger = logging.getLogger(__name__)


class GmailInboxEventHandler(abc.ABC):
    """
    A generic handler for all the events happening in the Gmail Inbox.
    """

    def on_message_added(self, event: events.MessageAddedEvent):
        """
        Handle the event when a new message is added to the Gmail Inbox.
        :param event: the event to handle
        """
        pass

    def on_message_deleted(self, event: events.MessageDeletedEvent):
        """
        Handle the event when a message is deleted from the Gmail Inbox.
        :param event: the event to handle
        """
        pass


class AgenticAutoReplyHandler(GmailInboxEventHandler):
    """
    An event handler that sends an automatic reply to the sender of the email.
    """

    def __init__(
        self,
        embedder_config: dict,
        qdrant_location: str,
        qdrant_api_key: Optional[str] = None,
    ):
        self.crew = AutoResponderCrew(
            embedder_config, qdrant_location, qdrant_api_key
        ).crew()

    def on_message_added(self, event: events.MessageAddedEvent):
        """
        Handle the event when a new message is added to the Gmail Inbox.
        :param event: the event to handle
        """
        service = event.service()
        message = event.message()
        logger.info("Received a new message: %s", message)

        # Load the full thread
        thread = service.load_full_thread(message.thread_id)
        last_message = thread.messages[-1]

        # We only want to process unread messages, so we skip the read ones
        if "UNREAD" not in last_message.label_ids:
            logger.info("The last message is already read. Skipping the reply.")
            return

        # If the last message is already a draft, then do not make a reply
        if "DRAFT" in last_message.label_ids:
            logger.info("The last message is already a draft. Skipping the reply.")
            return

        # Decode the messages and convert them to Markdown
        decoded_messages = [
            service.decode_message(message) for message in thread.messages
        ]
        md_messages = [
            markdownify(decoded_message.content) for decoded_message in decoded_messages
        ]

        # Call the crew to generate a response
        response = self.crew.kickoff(inputs={"messages": md_messages})
        logger.info("Generated response: %s", response.pydantic)
        if not isinstance(response.pydantic, models.EmailResponse):
            logger.info(
                "Crew decided not to respond to the message: %s", response.pydantic
            )
            return

        # Create a draft with the generated response
        email_response = response.pydantic
        if email_response.content is None:
            logger.info("The response is empty. Skipping the reply.")
            return

        service.add_draft(thread, content=email_response.content)
