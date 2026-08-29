import hashlib
import logging
import time
from pathlib import Path

import yaml
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)

from email_assistant import models
from email_assistant.crew import KnowledgeOrganizingCrew

logger = logging.getLogger(__name__)

# Watchdog fires several on_modified events for a single save (metadata +
# content writes), and Obsidian itself rewrites files multiple times within a
# second. Debounce them into a single re-ingestion.
MODIFY_DEBOUNCE_SEC = 2.0

# Obsidian (or sync tools) can hold a file locked for a moment while writing.
# Retry reads a few times before giving up.
FILE_READ_ATTEMPTS = 5
FILE_READ_RETRY_DELAY_SEC = 0.5


class AgenticObsidianVaultToQdrantHandler(FileSystemEventHandler):
    """
    An event handler for the changes done in the Obsidian Vault. It handles the
    synchronization between the filesystem and the Qdrant knowledge base used
    by the crew.
    """

    def __init__(
        self,
        embedder_config: dict,
        qdrant_location: str,
        qdrant_api_key: str | None = None,
        min_content_length: int = 10,
        sparse_embedder=None,
    ):
        crew = KnowledgeOrganizingCrew(
            embedder_config, qdrant_location, qdrant_api_key,
            sparse_embedder=sparse_embedder,
        )
        self.crew = crew.crew()
        self.knowledge_base = crew.knowledge_base()
        self.min_content_length = min_content_length
        self._last_modified_at: dict[str, float] = {}

    @staticmethod
    def _read_bytes_with_retry(path: Path) -> bytes:
        """
        Read the raw file bytes, retrying briefly when the file is locked by
        the writer (Windows raises PermissionError mid-write).
        """
        for attempt in range(1, FILE_READ_ATTEMPTS + 1):
            try:
                return path.read_bytes()
            except PermissionError:
                if attempt == FILE_READ_ATTEMPTS:
                    raise
                logger.warning(
                    "File locked (%s), retry %d/%d in %.1fs...",
                    path, attempt, FILE_READ_ATTEMPTS, FILE_READ_RETRY_DELAY_SEC,
                )
                time.sleep(FILE_READ_RETRY_DELAY_SEC)

    @classmethod
    def _file_hash(cls, path: Path) -> str:
        """Return the SHA-256 hash of the file's raw bytes."""
        return hashlib.sha256(cls._read_bytes_with_retry(path)).hexdigest()

    @classmethod
    def _read_text_with_retry(cls, path: Path) -> str:
        """Read the file text (UTF-8 with BOM tolerance), retrying on locks."""
        for attempt in range(1, FILE_READ_ATTEMPTS + 1):
            try:
                return path.read_text(encoding="utf-8-sig", errors="replace")
            except PermissionError:
                if attempt == FILE_READ_ATTEMPTS:
                    raise
                logger.warning(
                    "File locked (%s), retry %d/%d in %.1fs...",
                    path, attempt, FILE_READ_ATTEMPTS, FILE_READ_RETRY_DELAY_SEC,
                )
                time.sleep(FILE_READ_RETRY_DELAY_SEC)

    def initialize(self, init_path: Path):
        """
        Initialize the Qdrant collection with existing files.
        """
        self._cleanup_orphans(init_path)

        for file_path in Path(init_path).rglob("*.md"):
            file_path_str = str(file_path)
            file_hash = self._file_hash(file_path)
            # Count chunks stored for this exact content hash. Comparing the
            # count against total_chunks (rather than > 0) detects partially
            # written files from an interrupted previous ingestion.
            file_filter = {"src_path": file_path_str, "content_hash": file_hash}
            points_count = self.knowledge_base.count(file_filter)
            expected = self.knowledge_base.get_metadata_value(
                file_filter, "total_chunks"
            )
            if points_count > 0 and expected is not None and points_count >= expected:
                logger.info("File unchanged, skipping: %s", file_path)
                continue
            if points_count > 0:
                logger.info(
                    "File partially ingested (%d/%s chunks), re-ingesting: %s",
                    points_count, expected, file_path,
                )

            self.on_created(
                FileCreatedEvent(file_path_str, file_path_str, is_synthetic=True)
            )

    def _cleanup_orphans(self, init_path: Path):
        """
        Remove points whose src_path no longer exists in the vault, so that
        deleting a note in Obsidian while the app is stopped is reflected on
        the next start.
        """
        indexed_paths = self.knowledge_base.list_src_paths()
        vault_paths = {str(p) for p in Path(init_path).rglob("*.md")}
        for path in indexed_paths - vault_paths:
            logger.info("Removing orphaned entries for deleted file: %s", path)
            self.knowledge_base.delete({"src_path": path})

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        """
        Load a new file content into Qdrant knowledge base. Ignore new directories.
        :param event:
        :return:
        """
        if isinstance(event, DirCreatedEvent):
            return

        # Verify if the file is really a Markdown file
        if not event.src_path.endswith(".md"):
            return

        # Skip files inside Obsidian's trash folder (deleted notes).
        if "/.trash/" in event.src_path:
            return

        # Log the event
        logger.info("New file created: %s", event.src_path)

        # Load the file content. UTF-8 with sig handles BOM; errors="replace"
        # avoids crashing the watchdog thread on partially-written files.
        file_content = self._read_text_with_retry(Path(event.src_path)).strip()

        # Only process the file if the content is longer than the minimum length
        if len(file_content) < self.min_content_length:
            logger.info(
                "The file content is shorter than the minimum length of %i: %s",
                self.min_content_length,
                event.src_path,
            )
            return

        # Load the frontmatter from the Markdown file. Only files that start
        # with the YAML frontmatter marker ("---") are parsed; plain Markdown
        # files (without frontmatter) are skipped instead of failing to parse.
        try:
            if file_content.startswith("---"):
                frontmatter = next(yaml.safe_load_all(file_content))
                if not isinstance(frontmatter, dict):
                    frontmatter = {}
            else:
                frontmatter = {}
        except (StopIteration, yaml.YAMLError):
            frontmatter = {}

        # Skip Excalidraw drawings — their .md files hold compressed drawing data
        # (large base64 blobs), not prose notes, and blow up the LLM request.
        if frontmatter.get("excalidraw-plugin") is not None:
            logger.info("Skipping Excalidraw drawing: %s", event.src_path)
            return

        # Run the knowledge organizing crew to store the file content in the knowledge base.
        # Retry on transient gateway errors (e.g. 502) so a single failure does not
        # abort the whole ingestion run.
        response = None
        max_attempts = 5
        retry_delays = (10, 20, 30, 60)  # seconds, growing backoff
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.crew.kickoff(
                    inputs={"src_path": event.src_path, "document": file_content}
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt == max_attempts:
                    logger.error(
                        "Failed to process %s after %d attempts, skipping: %s",
                        event.src_path,
                        max_attempts,
                        e,
                    )
                    return
                delay = retry_delays[attempt - 1]
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                    attempt,
                    max_attempts,
                    event.src_path,
                    e,
                    delay,
                )
                time.sleep(delay)
        if not isinstance(response.pydantic, models.ContextualizedChunks):
            logger.info("Did not receive any contextualized chunks: %s", response)
            return

        # Store the response in the Qdrant knowledge base. Remove any existing
        # entries for this file first, then write the new chunks tagged with the
        # file's content hash so the next run can detect unchanged files.
        file_hash = self._file_hash(Path(event.src_path))
        self.knowledge_base.delete({"src_path": event.src_path})
        document_chunks: models.ContextualizedChunks = response.pydantic  # noqa
        total_chunks = len(document_chunks.chunks)
        batch: list[tuple[str, dict]] = []
        for chunk in document_chunks.chunks:
            formatted_input_data = f"{chunk.content}\n\n{chunk.context}"
            metadata = {
                "src_path": event.src_path,
                "content_hash": file_hash,
                "total_chunks": total_chunks,
                "chunk_context": chunk.context,
                "chunk_content": chunk.content,
                **frontmatter,
            }
            batch.append((formatted_input_data, metadata))
        self.knowledge_base.save_batch(batch)

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        """
        When the file is removed, all its content has to be removed from Qdrant as well.
        If a directory is removed, there is a separate event triggered for all the files inside, so
        we ignore directories.
        :param event:
        :return:
        """
        if isinstance(event, DirDeletedEvent):
            return

        # Verify if the file is really a Markdown file
        if not event.src_path.endswith(".md"):
            return

        # Log the event
        logger.info("File deleted: %s", event.src_path)

        # Remove all the entries related to the file from the knowledge base
        self.knowledge_base.delete({"src_path": event.src_path})

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        """
        When the file is modified, remove all the existing content related to
        this file in Qdrant, and then load the new content. Ignore directories,
        as modifications to directories themselves do not mean anything in
        terms of the content.
        :param event:
        :return:
        """
        if isinstance(event, DirModifiedEvent):
            return

        # Verify if the file is really a Markdown file
        if not event.src_path.endswith(".md"):
            return

        # Debounce: editors and sync tools fire multiple modified events for
        # a single save. Re-ingesting is expensive (LLM calls), so collapse
        # events that arrive within the debounce window.
        now = time.monotonic()
        last = self._last_modified_at.get(event.src_path)
        if last is not None and (now - last) < MODIFY_DEBOUNCE_SEC:
            self._last_modified_at[event.src_path] = now
            return
        self._last_modified_at[event.src_path] = now

        # Log the event
        logger.info("File modified: %s", event.src_path)

        # Remove the existing content
        self.on_deleted(
            FileDeletedEvent(event.src_path, event.dest_path, is_synthetic=True)
        )

        # Load the new content
        self.on_created(
            FileCreatedEvent(event.src_path, event.dest_path, is_synthetic=True)
        )

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        """
        Update the file path in Qdrant knowledge base. Ignore directories.
        :param event:
        :return:
        """
        if isinstance(event, DirMovedEvent):
            return

        # Verify if the file is really a Markdown file
        if not event.src_path.endswith(".md"):
            return

        # Log the event
        logger.info("File moved: %s -> %s", event.src_path, event.dest_path)

        # Remove the existing content from the old path
        self.on_deleted(
            FileDeletedEvent(event.src_path, event.dest_path, is_synthetic=True)
        )

        # Load the new content from the new location
        self.on_created(
            FileCreatedEvent(event.dest_path, event.dest_path, is_synthetic=True)
        )
