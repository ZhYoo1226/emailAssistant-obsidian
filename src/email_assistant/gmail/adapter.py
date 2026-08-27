import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import lru_cache
from pathlib import Path
from typing import Optional, Generator

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from email_assistant.gmail import models

logger = logging.getLogger(__name__)


class GmailServiceAdapter:
    """
    An adapter over the Gmail API service to simplify the interactions with it and
    use structured data classes instead of the raw JSON objects.
    """

    CREDENTIALS_FILE_NAME = "credentials.json"
    TOKEN_FILE_NAME = "token.json"
    GOOGLE_API_SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.modify",
    ]
    DEFAULT_CHARSET = "utf-8"
    CONTENT_TYPE_PREFERRED = ["text/html", "text/plain"]

    def __init__(self, credentials_dir: Path):
        self._credentials_dir = credentials_dir
        if not self._credentials_dir.exists():
            self._credentials_dir.mkdir(parents=True)
        self._credentials: Optional[Credentials] = None
        self._service: Optional[Resource] = None

    def is_authenticated(self) -> bool:
        """
        Check if the user is authenticated in the Google API.
        :return:
        """
        return self._service is not None

    def authenticate(self):
        """
        Authenticate the user with the Gmail API. It should redirect the user to the
        Google login page to authorize the application, if the user has not authorized
        the application yet.
        :return:
        """
        token_file = self._credentials_dir / self.TOKEN_FILE_NAME
        if token_file.exists():
            self._credentials = Credentials.from_authorized_user_file(
                str(token_file), self.GOOGLE_API_SCOPES
            )

        # User has to log in, because we do not have valid _credentials
        credentials_file = self._credentials_dir / self.CREDENTIALS_FILE_NAME
        if self._credentials is None or not self._credentials.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_file), self.GOOGLE_API_SCOPES
            )
            self._credentials = flow.run_local_server(port=0)
        else:
            self._credentials.refresh(Request())

        # Save the _credentials for the next run
        with open(token_file, "w") as fp:
            fp.write(self._credentials.to_json())

        # Connect to Gmail Service
        self._service = build("gmail", "v1", credentials=self._credentials)

    def iter_unread_threads(self) -> Generator[models.Thread, None, None]:
        """
        Iterate over all the unread threads in the Gmail Inbox.
        :return: a generator of the unread threads
        """
        # Iterate over all the pages of the unread threads
        page_token = None
        while True:
            response = (
                self._service.users()
                .threads()
                .list(userId="me", q="is:unread", pageToken=page_token)
                .execute()
            )
            for thread_descriptor in response.get("threads", []):
                full_thread = self.load_full_thread(thread_descriptor["id"])
                yield full_thread
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    def iter_history(
        self, last_history_id: int
    ) -> Generator[models.History, None, None]:
        """
        Iterate over the history of the Gmail Inbox starting from the given history ID.
        :param last_history_id: the history ID to start from
        :return: a generator of the history objects
        """
        # Iterate over all the pages of the history
        page_token = None
        while True:
            try:
                response = (
                    self._service.users()
                    .history()
                    .list(
                        userId="me",
                        startHistoryId=str(last_history_id),
                        pageToken=page_token,
                        historyTypes=["messageAdded", "messageDeleted"],
                    )
                    .execute()
                )
                for history_descriptor in response.get("history", []):
                    full_history = self._load_full_history(history_descriptor)
                    yield full_history
                page_token = response.get("nextPageToken")
                if not page_token:
                    logger.info("No more pages to load.")
                    break
            except TimeoutError:
                logger.error("Timeout error occurred. Retrying...")
                continue
            except HttpError as e:
                logger.error("HTTP error occurred: %s", e)
                return

    def load_max_history_id(self) -> Optional[int]:
        """
        Load the maximum history ID from the Gmail service. It loads just the last
        message and takes its history ID.
        :return: the maximum history ID
        """
        messages = (
            self._service.users().messages().list(userId="me", maxResults=1).execute()
        )
        if not messages.get("messages", []):
            return None
        message_id = messages["messages"][0]["id"]
        last_message = (
            self._service.users().messages().get(userId="me", id=message_id).execute()
        )
        return int(last_message["historyId"])

    def load_full_thread(self, thread_id: str) -> models.Thread:
        """
        Load the full thread from the Gmail service.
        :param thread_id: the ID of the thread to load
        :return: the full thread object
        """
        full_thread = (
            self._service.users()
            .threads()
            .get(userId="me", id=thread_id, format="full")
            .execute()
        )
        return models.Thread(**full_thread)

    @lru_cache
    def load_full_message(self, message_id: str) -> models.Message:
        """
        Load the full message from the Gmail service.
        :param message_id: the ID of the message to load
        :return: the full message object
        """
        full_message = (
            self._service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return models.Message(**full_message)

    def decode_message(self, message: models.Message) -> models.DecodedMessage:
        """
        Decode the message content from the base64 encoding.
        :param message:
        :return:
        """
        content = self._extract_message_content(message)
        return models.DecodedMessage(message=message, content=content)

    def add_draft(self, thread: models.Thread, content: str):
        """
        Add a draft message to the thread. It accepts the HTML content of the message
        and converts it to plain text internally.
        :param thread:
        :param content:
        :return:
        """
        # Last message is used to get all the metadata
        last_message = thread.messages[-1]

        # Remove HTML to create a plain text version
        plain_content = BeautifulSoup(content, "html.parser").get_text()

        # Build the email message
        email_message = MIMEMultipart("alternative")
        email_message["To"] = self._parse_email(last_message.get_header_value("From"))
        email_message["From"] = self._parse_email(last_message.get_header_value("To"))
        email_message["Subject"] = last_message.get_header_value("Subject")
        email_message["In-Reply-To"] = last_message.get_header_value("Message-ID")
        email_message["References"] = last_message.get_header_value("Message-ID")

        # Create the plain and HTML parts
        plain_part = MIMEText(plain_content, "plain")
        html_part = MIMEText(content, "html")

        # Attach the parts to the email message
        email_message.attach(plain_part)
        email_message.attach(html_part)

        # Create a dict-like object from the MIMEText object
        email_draft = {
            "message": {
                "threadId": thread.id,
                "raw": base64.urlsafe_b64encode(
                    email_message.as_string().encode("utf-8")
                ).decode(),
            }
        }

        # Create the draft message
        draft = (
            self._service.users()
            .drafts()
            .create(userId="me", body=email_draft)
            .execute()
        )
        logger.info("Created a draft message: %s", draft)

    def _extract_message_content(self, message: models.Message) -> str:
        """
        Extract the message content from the message object.
        :param message:
        :return:
        """
        content: Optional[str] = None
        charset: str = self.DEFAULT_CHARSET
        for mime_type in self.CONTENT_TYPE_PREFERRED:
            # Convert to lowercase for comparison
            mime_type = mime_type.lower()

            # If the body has data, use it
            if (
                message.payload.mime_type.lower() == mime_type
                and message.payload.body.data
            ):
                content = message.payload.body.data
                charset = self._extract_content_charset(message.payload)
                logger.debug(f"Found {mime_type} body with charset {charset}")
                break

            # Some messages have no parts, so we need to check if the parts exist
            if not message.payload.parts:
                continue

            # If the body has no data, check the parts, but first flatten the list of parts
            flatten_parts = self._flatten_message_parts(message)
            for part in flatten_parts:
                if not part.mime_type.lower() == mime_type:
                    continue
                content = part.body.data
                charset = self._extract_content_charset(part)
                logger.debug(
                    f"Found {mime_type} part with charset {charset} in part {part.part_id}"
                )
                break

            # If content found, break the loop
            if content:
                break

        # If no text body found, raise an error
        if content is None:
            raise ValueError("No text body found.")

        # Content is base64-encoded, decode it
        base64_decoded = base64.urlsafe_b64decode(content)
        try:
            return base64_decoded.decode(charset)
        except LookupError:
            # Fallback to default if the charset is not found
            return base64_decoded.decode(self.DEFAULT_CHARSET)

    def _load_full_history(self, history_descriptor: dict) -> models.History:
        """
        Parse the history descriptor and load the full history object out of it.
        :param history_descriptor:
        :return:
        """
        if "messagesAdded" in history_descriptor:
            messages_added = []
            for message_added in history_descriptor["messagesAdded"]:
                try:
                    message = self.load_full_message(message_added["message"]["id"])
                    messages_added.append(models.MessageAdded(message=message))
                except Exception:  # noqa
                    logger.error("Failed to load the full message: %s", message_added)
            history_descriptor["messagesAdded"] = messages_added
        full_history = models.History(**history_descriptor)
        return full_history

    def _flatten_message_parts(
        self, message: models.Message
    ) -> list[models.MessagePart]:
        """
        Flatten the list of message parts into a single list.
        :param message:
        :return:
        """
        parts = [message.payload]
        for part in message.payload.parts or []:
            parts.extend(self._flatten_parts(part))
        return parts

    def _flatten_parts(self, part: models.MessagePart) -> list[models.MessagePart]:
        """
        Flatten the inner parts of the message part.
        :param part:
        :return:
        """
        parts = [part]
        for inner_part in part.parts or []:
            parts.extend(self._flatten_parts(inner_part))
        return parts

    def _extract_content_charset(self, message: models.Message) -> str:
        """
        Extract the charset from the Content-Type header.
        :param message:
        :return:
        """
        content_type_header = next(
            (
                header
                for header in message.headers
                if header.name.lower() == "content-type"
            ),
            None,
        )
        if content_type_header:
            content_type = content_type_header.value
            charset_index = content_type.find("charset=")
            charset_index_end = content_type.find(";", charset_index)
            if charset_index_end == -1:
                charset_index_end = len(content_type)
            charset = content_type[charset_index + len("charset=") : charset_index_end]
            return charset
        return self.DEFAULT_CHARSET

    def _parse_email(self, text: str) -> str:
        """
        Parse the email content from the formatted text like '"John Done" <johndone@email.com>',
        so it only returns the email address. If the text is already an email address, it returns it.
        :param text:
        :return:
        """
        if "<" in text and ">" in text:
            return text.split("<")[1].split(">")[0]
        return text
