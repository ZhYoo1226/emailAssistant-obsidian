import logging
import sys
from pathlib import Path
from typing import Any

from watchdog.observers import Observer

# Windows 控制台默认 GBK 编码；CrewAI 打印的 agent 日志里含有 © 之类的字符，
# 会让 print() 抛 UnicodeEncodeError。强制 UTF-8 并允许替换，保证一条日志
# 永远不会中断整个邮件处理流程。
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]

import config
from email_assistant.embeddings import (
    BgeRerankFunction,
    FastEmbedFunction,
    JiebaBM25Function,
)
from email_assistant.gmail.handlers import AgenticAutoReplyHandler
from email_assistant.gmail.inbox import GmailInboxListener, GmailInboxState
from email_assistant.obsidian.handlers import AgenticObsidianVaultToQdrantHandler

# 可选：提供了 API key 时初始化 AgentOps。crewai>=1.15 不再自带 agentops
# extra，所以这个包本身也变成了可选依赖。
try:
    import agentops  # pyright: ignore[reportMissingImports]
except ImportError:
    agentops = None

WORKING_DIR = Path(__file__).parent
GMAIL_INBOX_STATE_FILE = WORKING_DIR / "gmail_inbox_state.json"

logger = logging.getLogger(__name__)

if config.agentops_api_key is not None:
    if agentops is None:
        logger.warning("AGENTOPS_API_KEY is set but agentops is not installed.")
    else:
        agentops.init(api_key=config.agentops_api_key)


# 组装处：从 config（定义者）取模型名，在这里实例化检索服务
# （实现者是 email_assistant.embeddings）。三个 ONNX 模型共享一份实例，
# 文件监听和邮件回复两条链路不会各建一份。
embedder = FastEmbedFunction(config.embedding_model_name)
sparse_embedder = JiebaBM25Function(config.sparse_model_name)
reranker = BgeRerankFunction(config.reranker_model_name)


def create_filesystem_listener() -> Any:
    """
    监视 Obsidian 仓库中的任何变更，并将其加载进知识库。
    """
    obsidian_vault_path = config.obsidian_vault_path
    assert obsidian_vault_path is not None, "OBSIDIAN_VAULT_PATH must be set"
    logger.info("Watching for filesystem changes at %s", obsidian_vault_path)

    # 文件被创建、修改或删除时，会触发 handler 的对应方法
    event_handler = AgenticObsidianVaultToQdrantHandler(
        {"provider": embedder},
        config.qdrant_location,
        config.qdrant_api_key,
        sparse_embedder=sparse_embedder,
        vault_root=Path(obsidian_vault_path),
        include_folders=config.obsidian_include_folders,
        exclude_folders=config.obsidian_exclude_folders,
        exclude_frontmatter=config.obsidian_exclude_frontmatter,
    )

    # 用已有文件初始化 Qdrant 集合
    event_handler.initialize(Path(obsidian_vault_path))

    # Observer 负责监听文件系统事件
    listener = Observer()
    listener.schedule(event_handler, obsidian_vault_path, recursive=True)
    return listener


def create_gmail_listener() -> GmailInboxListener:
    """
    监视 Gmail 收件箱中的新邮件，并进行相应处理。
    """
    logger.info("Monitoring mailbox for new emails...")

    # 加载上次保存的收件箱状态
    if GMAIL_INBOX_STATE_FILE.exists():
        gmail_state = GmailInboxState.load_state(GMAIL_INBOX_STATE_FILE)
    else:
        # 默认情况下，监听器会处理过去所有未读会话。
        # 如果只想处理新会话，请设置 process_all_unread_threads=False。
        gmail_state = GmailInboxState(process_all_unread_threads=True)

    # 创建 agent 自动回复处理器
    auto_reply_handler = AgenticAutoReplyHandler(
        {"provider": embedder},
        config.qdrant_location,
        config.qdrant_api_key,
        sparse_embedder=sparse_embedder,
        reranker=reranker,
    )

    # 启动监听器监视邮箱。监听器会周期性持久化状态文件，
    # 因此崩溃也不会丢失进度。
    listener = GmailInboxListener(
        WORKING_DIR,
        state=gmail_state,
        polling_time_sec=60,  # 每分钟轮询一次
        state_file=GMAIL_INBOX_STATE_FILE,
    )
    listener.add_handler(auto_reply_handler)
    return listener


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 开始监视 Gmail 收件箱和文件系统变更
    logger.info("Starting the monitoring of Gmail inbox and filesystem changes")

    # 先连接 Obsidian 仓库并监视变更
    file_system_listener = create_filesystem_listener()
    file_system_listener.start()

    # 监视 Gmail 收件箱
    gmail_inbox_listener = create_gmail_listener()
    gmail_inbox_listener.start()

    # 等待所有线程结束（它们应无限运行，直到被中断）
    try:
        file_system_listener.join()
        gmail_inbox_listener.join()
    except KeyboardInterrupt:
        logger.info("Stopping the monitoring of the filesystem and Gmail inbox...")

        file_system_listener.stop()
        gmail_inbox_listener.stop()

        # 保存 Gmail 收件箱状态
        gmail_inbox_listener.state().save(GMAIL_INBOX_STATE_FILE)

        logger.info("Monitoring stopped! Exiting.")
        sys.exit(0)
