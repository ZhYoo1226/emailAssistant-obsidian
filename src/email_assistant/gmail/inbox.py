import json
import logging
import time
from pathlib import Path
from typing import Optional
from google.oauth2.credentials import Credentials
from pydantic import BaseModel, Field
from watchdog.utils import BaseThread

from email_assistant.gmail import models, events
from email_assistant.gmail.adapter import GmailServiceAdapter
from email_assistant.gmail.handlers import GmailInboxEventHandler

logger = logging.getLogger(__name__)


class GmailInboxState(BaseModel):
    """
    A state of the Gmail Inbox processing. It allows to resume the processing
    from the last known state after a restart.
    """

    process_all_unread_threads: bool = Field(
        default=False,
        description=(
            "Decide if all the unread threads should be processed. If set to True, the last message of each "
            "unread thread will be processed and emit a MessageAddedEvent."
        ),
    )
    last_history_id: Optional[int] = None

    def update_last_history_id(self, new_history_id: int) -> bool:
        """
        Update the last history ID to the new value, but only if it is greater
        than the current one.
        :param new_history_id: the new history ID
        :return: True if the last history ID was updated, False otherwise
        """
        if self.last_history_id is None or new_history_id > self.last_history_id:
            self.last_history_id = new_history_id
            return True
        return False

    @classmethod
    def load_state(cls, path: Path) -> "GmailInboxState":
        """
        Load the state of the listener from the specified path.
        :param path: the path to load the state
        :return: the loaded state
        """
        with open(path, "r") as f:
            state_json = json.load(f)
            return GmailInboxState(**state_json)

    def save(self, path: Path) -> None:
        """
        Save the current state of the listener to the specified path.
        :param path: the path to save the state
        """
        with open(path, "w") as f:
            state_json = self.model_dump(mode="json")
            f.write(json.dumps(state_json))


class GmailInboxListener(BaseThread):
    """
    A listener runs a loop that listens for new events in the Gmail Inbox and
    triggers the event handlers once the event occurs.
    """

    DEFAULT_CHARSET = "utf-8"
    CONTENT_TYPE_PREFERRED = ["text/html", "text/plain"]

    def __init__(
        self,
        credentials_dir: Path,
        state: Optional[GmailInboxState] = None,
        polling_time_sec: int = 1,
    ):
        super().__init__()
        self._credentials_dir = credentials_dir
        if not self._credentials_dir.exists():
            self._credentials_dir.mkdir(parents=True)
        self._credentials: Optional[Credentials] = None
        self._service: GmailServiceAdapter = GmailServiceAdapter(credentials_dir)
        self._state = GmailInboxState() if state is None else state
        self._handlers: list[GmailInboxEventHandler] = []
        self._polling_time_sec = polling_time_sec

    def add_handler(self, handler: GmailInboxEventHandler):
        """
        Add a new handler to the listener.
        :param handler: the handler to add
        """
        self._handlers.append(handler)

    def on_thread_start(self) -> None:
        # Ensure the user is authenticated in Google API
        if not self._service.is_authenticated():
            self._service.authenticate()

    def run(self) -> None:
        """
        Start the listener and run the loop that listens for new events in the Gmail Inbox.
        """
        # Load all the unread threads and process them if requested
        if self._state.process_all_unread_threads:
            counter = -1
            for counter, unread_thread in enumerate(
                self._service.iter_unread_threads()
            ):
                # Update the last history ID, so we do not process the same threads again
                self._state.update_last_history_id(int(unread_thread.history_id))
                if not unread_thread.messages:
                    continue

                # Emit only the last message of the thread
                self.emit_message_added_event(unread_thread.messages[-1])
                if counter % 100 == 99:
                    logger.info("Processed %i unread threads", counter + 1)

            # Log the number of processed unread threads
            logger.info("Processed all unread threads (%i)", counter + 1)

            # Update the state so the unread threads are not processed again
            self._state.process_all_unread_threads = False

        while True:
            # If we still don't have the last history ID, we just extract the last one
            # from the Google Gmail service and sta`rt processing from here
            if self._state.last_history_id is None:
                current_max_history_id = self._service.load_max_history_id()
                self._state.update_last_history_id(current_max_history_id)

            # Log the state before starting the loop over the history
            logger.info("Current state: %s", self._state)

            # Get the history starting from the last known history ID
            counter = -1
            history_generator = self._service.iter_history(self._state.last_history_id)
            for counter, history in enumerate(history_generator):
                # Update the last history ID, so we do not process the same history again
                self._state.update_last_history_id(int(history.id))

                # Iterate over the messages added and call the handlers
                for message_added in history.messages_added:
                    self.emit_message_added_event(message_added.message)

                # Iterate over the messages deleted and call the handlers
                for message_deleted in history.messages_deleted:
                    self.emit_message_deleted_event(message_deleted.message)

                if counter % 100 == 99:
                    logger.info("Processed %i history events", counter + 1)

            # Log the number of processed history events
            logger.info("Processed %i history events", counter + 1)

            # Wait for the polling time to not overload the Gmail API
            time.sleep(self._polling_time_sec)

    def state(self) -> GmailInboxState:
        """
        Get the current state of the listener. Useful to save it to disk and load
        it after the restart.
        :return:
        """
        return self._state

    def emit_message_added_event(self, message: models.Message):
        """
        Emit the message added event to all the registered handlers.
        :param message: the message to emit
        """
        logger.debug("Emitting the message added event %s", message)
        event = events.MessageAddedEvent(self._service, message)
        for handler in self._handlers:
            try:
                handler.on_message_added(event)
            except Exception as e:
                logger.error("Error while handling the message added event: %s", event)
                logger.exception(e)

    def emit_message_deleted_event(self, message: models.Message):
        """
        Emit the message deleted event to all the registered handlers.
        :param message: the message to emit
        """
        logger.debug("Emitting the message deleted event %s", message.id)
        event = events.MessageDeletedEvent(self._service, message.id)
        for handler in self._handlers:
            try:
                handler.on_message_deleted(event)
            except Exception as e:
                logger.error(
                    "Error while handling the message deleted event: %s", event
                )
                logger.exception(e)
