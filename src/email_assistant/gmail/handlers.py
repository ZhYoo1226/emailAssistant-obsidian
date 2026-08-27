import abc
import logging
import os
from typing import Optional

from litellm import completion
from markdownify import markdownify

from email_assistant.gmail import events
from email_assistant import models
from email_assistant.crew import AutoResponderCrew

logger = logging.getLogger(__name__)

# Fast gateway model used for the faithfulness check.
GATEWAY_FAST_MODEL = os.environ.get("GATEWAY_FAST_MODEL", "deepseek-v4-flash")


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
        crew_builder = AutoResponderCrew(
            embedder_config, qdrant_location, qdrant_api_key
        )
        self.crew = crew_builder.crew()
        self.knowledge_base = crew_builder.knowledge_base()

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

        # Decode the messages and convert them to Markdown. Skip drafts — they
        # are the system's own previous replies and must not be fed back into
        # the categorizer (which would otherwise misfile the thread as spam).
        non_draft_messages = [
            message for message in thread.messages
            if "DRAFT" not in message.label_ids
        ]
        decoded_messages = [
            service.decode_message(message) for message in non_draft_messages
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

        # Verify citations before creating the draft: the cited sources must be
        # real files in the knowledge base, and the reply must be faithful to them.
        if not self._verify_sources(email_response):
            logger.warning("Sources could not be verified. Skipping the draft.")
            return
        if not self._verify_faithfulness(email_response):
            logger.warning("Response not faithful to sources. Skipping the draft.")
            return

        service.add_draft(thread, content=email_response.content)

    def _verify_sources(self, email_response: models.EmailResponse) -> bool:
        """
        Check that every cited src_path actually exists in the knowledge base.
        An empty sources list is allowed (e.g. "I cannot provide a response").
        """
        if not email_response.sources:
            return True
        indexed_paths = self.knowledge_base.list_src_paths()
        for source in email_response.sources:
            if source.src_path not in indexed_paths:
                logger.warning(
                    "Cited source not in knowledge base: %s", source.src_path
                )
                return False
        return True

    def _verify_faithfulness(self, email_response: models.EmailResponse) -> bool:
        """
        Ask the fast model whether the reply's factual statements are supported
        by the cited sources (a cheap LLM stand-in for a dedicated NLI model).
        """
        if not email_response.sources:
            return True
        sources_text = "\n".join(
            f"[{s.index}] {s.src_path}: {s.snippet}" for s in email_response.sources
        )
        prompt = (
            "You are checking whether an email reply is faithful to its cited sources.\n\n"
            f"Reply content:\n{email_response.content}\n\n"
            f"Cited sources:\n{sources_text}\n\n"
            "Question: is every factual statement in the reply supported by the cited "
            "sources? Answer with only YES or NO."
        )
        response = completion(
            model=f"openai/{GATEWAY_FAST_MODEL}",
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
        logger.info("Faithfulness verdict: %s", verdict)
        return verdict.startswith("YES")
