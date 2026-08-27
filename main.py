import logging
import signal
from pathlib import Path

from watchdog.observers import Observer as FileSystemListener

import config
from email_assistant.gmail.handlers import AgenticAutoReplyHandler
from email_assistant.gmail.inbox import GmailInboxListener, GmailInboxState
from email_assistant.obsidian.handlers import AgenticObsidianVaultToQdrantHandler

import agentops

# Set the default signal handler for SIGINT, so the KeyboardInterrupt exception is raised
signal.signal(signal.SIGINT, signal.default_int_handler)

# Optional: Initialize AgentOps if the API key is provided
if config.agentops_api_key is not None:
    agentops.init(api_key=config.agentops_api_key)

WORKING_DIR = Path(__file__).parent
GMAIL_INBOX_STATE_FILE = WORKING_DIR / "gmail_inbox_state.json"

logger = logging.getLogger(__name__)


def create_filesystem_listener() -> FileSystemListener:
    """
    Watch any changes done in the Obsidian vault and load them into the knowledge base.
    """
    logger.info("Watching for filesystem changes at %s", config.obsidian_vault_path)

    # Handler's methods are going to be called when a file is created, modified, or deleted
    event_handler = AgenticObsidianVaultToQdrantHandler(
        config.embedder_config, config.qdrant_location, config.qdrant_api_key
    )

    # Initialize the Qdrant collection with existing files
    event_handler.initialize(config.obsidian_vault_path)

    # Observer listens for filesystem events
    listener = FileSystemListener()
    listener.schedule(event_handler, config.obsidian_vault_path, recursive=True)
    return listener


def create_gmail_listener() -> GmailInboxListener:
    """
    Monitor the Gmail inbox for new emails and handle them accordingly.
    """
    logger.info("Monitoring mailbox for new emails...")

    # Load the previous state of the Gmail inbox
    if GMAIL_INBOX_STATE_FILE.exists():
        gmail_state = GmailInboxState.load_state(GMAIL_INBOX_STATE_FILE)
    else:
        # By default, our listener will process all the unread threads from the past.
        # Please set process_all_unread_threads=False if you want to process only new threads.
        gmail_state = GmailInboxState(process_all_unread_threads=True)

    # Create an agentic auto-reply handler
    auto_reply_handler = AgenticAutoReplyHandler(
        config.embedder_config, config.qdrant_location, config.qdrant_api_key
    )

    # Start the listener so that it can monitor the mailbox
    listener = GmailInboxListener(
        WORKING_DIR,
        state=gmail_state,
        polling_time_sec=60,  # We poll every minute
    )
    listener.add_handler(auto_reply_handler)
    return listener


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Start monitoring the Gmail inbox and filesystem changes
    logger.info("Starting the monitoring of Gmail inbox and filesystem changes")

    # Connect to Obsidian vault first and monitor the changes
    file_system_listener = create_filesystem_listener()
    file_system_listener.start()

    # Monitor the Gmail inbox
    gmail_inbox_listener = create_gmail_listener()
    gmail_inbox_listener.start()

    # Wait until all threads are finished (they should run indefinitely, or until interrupted)
    try:
        file_system_listener.join()
        gmail_inbox_listener.join()
    except KeyboardInterrupt as e:
        logger.info("Stopping the monitoring of the filesystem and Gmail inbox...")

        file_system_listener.stop()
        gmail_inbox_listener.stop()

        # Save the state of the Gmail inbox
        gmail_inbox_listener.state().save(GMAIL_INBOX_STATE_FILE)

        logger.info("Monitoring stopped! Exiting.")
