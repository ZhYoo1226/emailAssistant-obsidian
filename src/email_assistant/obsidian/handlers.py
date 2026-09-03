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

# Watchdog 对一次保存会触发多个 on_modified 事件（元数据 + 内容写入），
# Obsidian 自己也会在一秒内多次重写文件。用防抖把它们合并成一次重新摄取。
MODIFY_DEBOUNCE_SEC = 2.0

# Obsidian（或同步工具）写入时可能短暂锁住文件。放弃前先重试几次读取。
FILE_READ_ATTEMPTS = 5
FILE_READ_RETRY_DELAY_SEC = 0.5

_T = TypeVar("_T")


def _read_with_retry(read: Callable[[Path], _T], path: Path) -> _T:
    """
    执行一次文件读取，文件被写入方短暂锁住时（Windows 会在写入中途抛
    PermissionError）做短暂重试。
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
    Markdown 文件事件的路径；目录或非 Markdown 事件返回 None。所有
    on_* 处理器共用的守卫——目录会产生各自的逐文件事件，而只有
    .md 文件才是知识。
    """
    if event.is_directory or not str(event.src_path).endswith(".md"):
        return None
    return os.fsdecode(event.src_path)


class AgenticObsidianVaultToQdrantHandler(FileSystemEventHandler):
    """
    处理 Obsidian 仓库变更的事件处理器。负责文件系统与 crew 使用的
    Qdrant 知识库之间的同步。
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
        orphan_protect_folders: list[str] | None = None,
    ):
        # CrewBase 的 TYPE_CHECKING stub 把 __init__ 标注为
        # (*args, **kwargs)，向 pyright 隐藏了 BaseCrew 真实的
        # 位置参数签名。
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
        # 共库保护：这些 metadata.folder 下的点不归本 handler 管理，
        # 孤儿清理直接跳过（由 .env 的 ORPHAN_PROTECT_FOLDERS 提供）。
        self._orphan_protect_folders = set(orphan_protect_folders or [])
        self._last_modified_at: dict[str, float] = {}

    def _top_level_folder(self, src_path: str) -> str | None:
        """
        文件所在的仓库顶层文件夹名；文件直接位于仓库根目录时返回
        None。仓库预期只使用一层文件夹（更深的嵌套不加以区分）。
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
        文件是否属于被选取用于摄取的文件夹。直接位于仓库根目录的
        文件始终在范围内。
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
        """读取文件原始字节，写入方短暂锁住时重试。"""
        return _read_with_retry(Path.read_bytes, path)

    @classmethod
    def _file_hash(cls, path: Path) -> str:
        """返回文件原始字节的 SHA-256 哈希。"""
        return hashlib.sha256(cls._read_bytes_with_retry(path)).hexdigest()

    @staticmethod
    def _read_text_with_retry(path: Path) -> str:
        """读取文件文本（UTF-8 且容忍 BOM），锁住时重试。"""

        def _read(p: Path) -> str:
            return p.read_text(encoding="utf-8-sig", errors="replace")

        return _read_with_retry(_read, path)

    def initialize(self, init_path: Path):
        """
        用已有文件初始化 Qdrant 集合。
        """
        vault_md_files = [p for p in Path(init_path).rglob("*.md")]
        self._cleanup_orphans(vault_md_files)

        for file_path in vault_md_files:
            file_path_str = str(file_path)
            if not self._in_scope(file_path_str):
                logger.info("Out of configured folders, skipping: %s", file_path)
                continue
            file_hash = self._file_hash(file_path)
            # 统计该内容哈希下已存储的 chunk 数。与 total_chunks 比较
            # （而不是 > 0）可以检测出上次摄取被中断导致的文件只写入了
            # 一部分。
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
        移除 src_path 已不存在于仓库中、或已不在配置文件夹范围内的点，
        这样应用停止期间的删除和范围变更会在下次启动时反映出来。

        受保护 folder（self._orphan_protect_folders）的点不属于本仓库
        摄取范围，不参与孤儿判定，绝不会被清理。
        """
        vault_paths = {
            str(p) for p in vault_md_files if self._in_scope(str(p))
        }
        stored_paths = self.knowledge_base.list_src_paths(
            exclude_folders=sorted(self._orphan_protect_folders)
        )
        for path in stored_paths - vault_paths:
            logger.info("Removing orphaned entries for deleted file: %s", path)
            self.knowledge_base.delete({"src_path": path})

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        """
        把新文件内容加载进 Qdrant 知识库。忽略新目录。
        :param event:
        :return:
        """
        src_path = _file_md_path(event)
        if src_path is None:
            return

        # 跳过配置排除文件夹之外的文件（典型如 .trash 回收站和
        # .obsidian 程序配置目录，见 .env 的 OBSIDIAN_EXCLUDE_FOLDERS）。
        if not self._in_scope(src_path):
            logger.info("Out of configured folders, skipping: %s", src_path)
            return

        # 记录事件
        logger.info("New file created: %s", src_path)

        # 读取文件内容。UTF-8 with sig 处理 BOM；errors="replace" 避免
        # 写到一半的文件让 watchdog 线程崩溃。
        file_content = self._read_text_with_retry(Path(src_path)).strip()

        # 只有内容超过最小长度的文件才处理
        if len(file_content) < self.min_content_length:
            logger.info(
                "The file content is shorter than the minimum length of %i: %s",
                self.min_content_length,
                src_path,
            )
            return

        # 从 Markdown 文件读取 frontmatter。只有以 YAML frontmatter 标记
        # （"---"）开头的文件才解析；普通 Markdown 文件（没有
        # frontmatter）直接跳过而不是解析失败。
        try:
            if file_content.startswith("---"):
                frontmatter = next(yaml.safe_load_all(file_content))
                if not isinstance(frontmatter, dict):
                    frontmatter = {}
            else:
                frontmatter = {}
        except (StopIteration, yaml.YAMLError):
            frontmatter = {}

        # 跳过配置排除键命中的笔记：如 Excalidraw 画板存的是压缩的
        # base64 数据块而不是正文，会把 LLM 请求撑爆（排除键来自 .env
        # 的 OBSIDIAN_EXCLUDE_FRONTMATTER）。
        if self._exclude_frontmatter & frontmatter.keys():
            logger.info(
                "Excluded by frontmatter key (%s), skipping: %s",
                self._exclude_frontmatter & frontmatter.keys(),
                src_path,
            )
            return

        # 运行知识整理 crew，把文件内容存进知识库。
        # 瞬时网关错误（如 502）时重试，单次失败不会中止整个摄取流程。
        response = None
        max_attempts = 5
        retry_delays = (10, 20, 30, 60)  # 秒，递增退避
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

        # 把响应存入 Qdrant 知识库。先移除该文件已有的条目，再写入
        # 带有文件内容哈希的新 chunk，下次运行就能检测出未变更的文件。
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
        文件被删除时，其所有内容也必须从 Qdrant 中移除。目录被删除时，
        里面的每个文件会各自触发一个事件，所以忽略目录。
        :param event:
        :return:
        """
        src_path = _file_md_path(event)
        if src_path is None:
            return

        # 记录事件
        logger.info("File deleted: %s", src_path)

        # 从知识库中移除与该文件相关的所有条目
        self.knowledge_base.delete({"src_path": src_path})

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        """
        文件被修改时，先移除 Qdrant 中与该文件相关的所有已有内容，
        再加载新内容。忽略目录——对目录本身的修改在内容层面没有意义。
        :param event:
        :return:
        """
        src_path = _file_md_path(event)
        if src_path is None:
            return

        # 防抖：编辑器和同步工具对一次保存会触发多个 modified 事件。
        # 重新摄取的开销很大（LLM 调用），所以把防抖窗口内到达的事件
        # 合并掉。
        now = time.monotonic()
        last = self._last_modified_at.get(src_path)
        if last is not None and (now - last) < MODIFY_DEBOUNCE_SEC:
            self._last_modified_at[src_path] = now
            return
        self._last_modified_at[src_path] = now

        # 记录事件
        logger.info("File modified: %s", src_path)

        # 移除已有内容
        self.on_deleted(
            FileDeletedEvent(src_path, event.dest_path, is_synthetic=True)
        )

        # 加载新内容
        self.on_created(
            FileCreatedEvent(src_path, event.dest_path, is_synthetic=True)
        )

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        """
        更新 Qdrant 知识库中的文件路径。忽略目录。
        :param event:
        :return:
        """
        src_path = _file_md_path(event)
        if src_path is None:
            return

        # 记录事件
        logger.info("File moved: %s -> %s", event.src_path, event.dest_path)

        # 移除旧路径下的已有内容
        self.on_deleted(
            FileDeletedEvent(src_path, event.dest_path, is_synthetic=True)
        )

        # 从新位置加载新内容
        self.on_created(
            FileCreatedEvent(event.dest_path, event.dest_path, is_synthetic=True)
        )
