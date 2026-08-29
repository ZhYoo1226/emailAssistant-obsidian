import hashlib
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

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
    FileSystemEvent,
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

_T = TypeVar("_T")


def _read_with_retry(read: Callable[[Path], _T], path: Path) -> _T:
    """
    Run a file read, retrying briefly when the file is locked by the
    writer (Windows raises PermissionError mid-write).
    """
    for attempt in range(1, FILE_READ_ATTEMPTS + 1):
        try:
            return read(path)
        except PermissionError:
            if attempt == FILE_READ_ATTEMPTS:
                raise
            logger.warning(
                "File locked (%s), retry %d/%d in %.1fs...",
                path, attempt, FILE_READ_ATTEMPTS, FILE_READ_RETRY_DELAY_SEC,
            )
            time.sleep(FILE_READ_RETRY_DELAY_SEC)
    raise AssertionError("unreachable")  # pragma: no cover


def _file_md_path(event: FileSystemEvent) -> str | None:
    """
    Path of a Markdown file event, or None for directory / non-Markdown
    events. Shared guard for all on_* handlers — directories produce
    per-file events of their own, and only .md files are knowledge.
    """
    if event.is_directory or not str(event.src_path).endswith(".md"):
        return None
    return os.fsdecode(event.src_path)


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
        vault_root: Path | None = None,
        include_folders: list[str] | None = None,
        exclude_folders: list[str] | None = None,
        exclude_frontmatter: list[str] | None = None,
    ):
        # CrewBase's TYPE_CHECKING stub types __init__ as (*args, **kwargs),
        # hiding the real BaseCrew positional signature from pyright.
        crew = KnowledgeOrganizingCrew(
            embedder_config,  # pyright: ignore[reportCallIssue]
            qdrant_location,
            qdrant_api_key,
            sparse_embedder=sparse_embedder,
        )
        self.crew = crew.crew()
        self.knowledge_base = crew.knowledge_base()  # pyright: ignore[reportAttributeAccessIssue]
        self.min_content_length = min_content_length
        self._vault_root = vault_root
        self._include_folders = set(include_folders or [])
        self._exclude_folders = set(exclude_folders or [])
        self._exclude_frontmatter = set(exclude_frontmatter or [])
        self._last_modified_at: dict[str, float] = {}

    def _top_level_folder(self, src_path: str) -> str | None:
        """
        Name of the top-level vault folder containing the file, or None for
        files directly in the vault root. The vault is expected to use a single
        level of folders (deeper nesting is not distinguished).
        """
        if self._vault_root is None:
            return None
        try:
            relative = Path(src_path).relative_to(self._vault_root)
        except ValueError:
            return None
        parts = relative.parts
        return parts[0] if len(parts) > 1 else None

    def _in_scope(self, src_path: str) -> bool:
        """
        Whether a file belongs to a folder selected for ingestion. Files
        directly in the vault root are always in scope.
        """
        folder = self._top_level_folder(src_path)
        if folder is None:
            return True
        if folder in self._exclude_folders:
            return False
        if self._include_folders and folder not in self._include_folders:
            return False
        return True

    @staticmethod
    def _read_bytes_with_retry(path: Path) -> bytes:
        """Read the raw file bytes, retrying briefly on writer locks."""
        return _read_with_retry(Path.read_bytes, path)

    @classmethod
    def _file_hash(cls, path: Path) -> str:
        """Return the SHA-256 hash of the file's raw bytes."""
        return hashlib.sha256(cls._read_bytes_with_retry(path)).hexdigest()

    @staticmethod
    def _read_text_with_retry(path: Path) -> str:
        """Read the file text (UTF-8 with BOM tolerance), retrying on locks."""

        def _read(p: Path) -> str:
            return p.read_text(encoding="utf-8-sig", errors="replace")

        return _read_with_retry(_read, path)

    def initialize(self, init_path: Path):
        """
        Initialize the Qdrant collection with existing files.
        """
        vault_md_files = [p for p in Path(init_path).rglob("*.md")]
        self._cleanup_orphans(vault_md_files)

        for file_path in vault_md_files:
            file_path_str = str(file_path)
            if not self._in_scope(file_path_str):
                logger.info("Out of configured folders, skipping: %s", file_path)
                continue
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

    def _cleanup_orphans(self, vault_md_files: list[Path]):
        """
        Remove points whose src_path no longer exists in the vault or falls
        outside the configured folders, so deletions and scope changes made
        while the app is stopped are reflected on the next start.
        """
        vault_paths = {
            str(p) for p in vault_md_files if self._in_scope(str(p))
        }
        for path in self.knowledge_base.list_src_paths() - vault_paths:
            logger.info("Removing orphaned entries for deleted file: %s", path)
            self.knowledge_base.delete({"src_path": path})

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        """
        Load a new file content into Qdrant knowledge base. Ignore new directories.
        :param event:
        :return:
        """
        src_path = _file_md_path(event)
        if src_path is None:
            return

        # Skip files outside the configured vault folders (includes
        # Obsidian's .trash folder, which holds deleted notes).
        if not self._in_scope(src_path):
            logger.info("Out of configured folders, skipping: %s", src_path)
            return

        # Log the event
        logger.info("New file created: %s", src_path)

        # Load the file content. UTF-8 with sig handles BOM; errors="replace"
        # avoids crashing the watchdog thread on partially-written files.
        file_content = self._read_text_with_retry(Path(src_path)).strip()

        # Only process the file if the content is longer than the minimum length
        if len(file_content) < self.min_content_length:
            logger.info(
                "The file content is shorter than the minimum length of %i: %s",
                self.min_content_length,
                src_path,
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

        # Skip plugin-generated notes: e.g. Excalidraw drawings store
        # compressed base64 blobs, not prose, and blow up the LLM request.
        if self._exclude_frontmatter & frontmatter.keys():
            logger.info(
                "Excluded by frontmatter key (%s), skipping: %s",
                self._exclude_frontmatter & frontmatter.keys(),
                src_path,
            )
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
                    inputs={"src_path": src_path, "document": file_content}
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt == max_attempts:
                    logger.error(
                        "Failed to process %s after %d attempts, skipping: %s",
                        src_path,
                        max_attempts,
                        e,
                    )
                    return
                delay = retry_delays[attempt - 1]
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                    attempt,
                    max_attempts,
                    src_path,
                    e,
                    delay,
                )
                time.sleep(delay)
        pydantic_output = getattr(response, "pydantic", None)
        if not isinstance(pydantic_output, models.ContextualizedChunks):
            logger.info("Did not receive any contextualized chunks: %s", response)
            return

        # Store the response in the Qdrant knowledge base. Remove any existing
        # entries for this file first, then write the new chunks tagged with the
        # file's content hash so the next run can detect unchanged files.
        file_hash = self._file_hash(Path(src_path))
        self.knowledge_base.delete({"src_path": src_path})
        document_chunks: models.ContextualizedChunks = pydantic_output
        total_chunks = len(document_chunks.chunks)
        folder = self._top_level_folder(src_path)
        batch: list[tuple[str, dict]] = []
        for chunk in document_chunks.chunks:
            formatted_input_data = f"{chunk.content}\n\n{chunk.context}"
            metadata = {
                "src_path": src_path,
                "content_hash": file_hash,
                "total_chunks": total_chunks,
                "chunk_context": chunk.context,
                "chunk_content": chunk.content,
                **({"folder": folder} if folder else {}),
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
        src_path = _file_md_path(event)
        if src_path is None:
            return

        # Log the event
        logger.info("File deleted: %s", src_path)

        # Remove all the entries related to the file from the knowledge base
        self.knowledge_base.delete({"src_path": src_path})

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        """
        When the file is modified, remove all the existing content related to
        this file in Qdrant, and then load the new content. Ignore directories,
        as modifications to directories themselves do not mean anything in
        terms of the content.
        :param event:
        :return:
        """
        src_path = _file_md_path(event)
        if src_path is None:
            return

        # Debounce: editors and sync tools fire multiple modified events for
        # a single save. Re-ingesting is expensive (LLM calls), so collapse
        # events that arrive within the debounce window.
        now = time.monotonic()
        last = self._last_modified_at.get(src_path)
        if last is not None and (now - last) < MODIFY_DEBOUNCE_SEC:
            self._last_modified_at[src_path] = now
            return
        self._last_modified_at[src_path] = now

        # Log the event
        logger.info("File modified: %s", src_path)

        # Remove the existing content
        self.on_deleted(
            FileDeletedEvent(src_path, event.dest_path, is_synthetic=True)
        )

        # Load the new content
        self.on_created(
            FileCreatedEvent(src_path, event.dest_path, is_synthetic=True)
        )

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        """
        Update the file path in Qdrant knowledge base. Ignore directories.
        :param event:
        :return:
        """
        src_path = _file_md_path(event)
        if src_path is None:
            return

        # Log the event
        logger.info("File moved: %s -> %s", event.src_path, event.dest_path)

        # Remove the existing content from the old path
        self.on_deleted(
            FileDeletedEvent(src_path, event.dest_path, is_synthetic=True)
        )

        # Load the new content from the new location
        self.on_created(
            FileCreatedEvent(event.dest_path, event.dest_path, is_synthetic=True)
        )
