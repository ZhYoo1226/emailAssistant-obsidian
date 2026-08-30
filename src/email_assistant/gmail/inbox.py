import json
import logging
import time
from pathlib import Path

from google.oauth2.credentials import Credentials
from pydantic import BaseModel, Field
from watchdog.utils import BaseThread

from email_assistant.gmail import events, models
from email_assistant.gmail.adapter import GmailServiceAdapter, HistoryExpiredError
from email_assistant.gmail.handlers import GmailInboxEventHandler

logger = logging.getLogger(__name__)

# 处理期间每隔多少秒把状态刷到磁盘一次，这样崩溃也不会丢失自上次
# 优雅关闭以来的游标进度。
STATE_SAVE_INTERVAL_SEC = 60


class GmailInboxState(BaseModel):
    """
    Gmail 收件箱的处理状态。重启后可以从最后已知的状态恢复处理。
    """

    process_all_unread_threads: bool = Field(
        default=False,
        description=(
            "Decide if all the unread threads should be processed. If set to True, the "
            "last message of each unread thread will be processed and emit a "
            "MessageAddedEvent."
        ),
    )
    last_history_id: int | None = None

    def update_last_history_id(self, new_history_id: int) -> bool:
        """
        用新值更新最后的 history ID，但仅当新值大于当前值时才更新。
        :param new_history_id: 新的 history ID
        :return: 更新了则返回 True，否则返回 False
        """
        if self.last_history_id is None or new_history_id > self.last_history_id:
            self.last_history_id = new_history_id
            return True
        return False

    @classmethod
    def load_state(cls, path: Path) -> "GmailInboxState":
        """
        从指定路径加载监听器状态。
        :param path: 加载状态的路径
        :return: 加载的状态
        """
        with open(path, encoding="utf-8") as f:
            state_json = json.load(f)
            return GmailInboxState(**state_json)

    def save(self, path: Path) -> None:
        """
        把监听器当前状态保存到指定路径。
        :param path: 保存状态的路径
        """
        with open(path, "w", encoding="utf-8") as f:
            state_json = self.model_dump(mode="json")
            f.write(json.dumps(state_json))


class GmailInboxListener(BaseThread):
    """
    监听器运行一个循环，监听 Gmail 收件箱中的新事件，事件发生时
    触发对应的事件处理器。
    """

    DEFAULT_CHARSET = "utf-8"
    CONTENT_TYPE_PREFERRED = ["text/html", "text/plain"]

    def __init__(
        self,
        credentials_dir: Path,
        state: GmailInboxState | None = None,
        polling_time_sec: int = 1,
        state_file: Path | None = None,
    ):
        super().__init__()
        self._credentials_dir = credentials_dir
        if not self._credentials_dir.exists():
            self._credentials_dir.mkdir(parents=True)
        self._credentials: Credentials | None = None
        self._service: GmailServiceAdapter = GmailServiceAdapter(credentials_dir)
        self._state = GmailInboxState() if state is None else state
        self._handlers: list[GmailInboxEventHandler] = []
        self._polling_time_sec = polling_time_sec
        self._state_file = state_file
        self._last_state_save = time.monotonic()

    def add_handler(self, handler: GmailInboxEventHandler):
        """
        向监听器添加新的处理器。
        :param handler: 要添加的处理器
        """
        self._handlers.append(handler)

    def on_thread_start(self) -> None:
        # 确保用户已通过 Google API 认证
        if not self._service.is_authenticated():
            self._service.authenticate()

    def _save_state_if_due(self, force: bool = False) -> None:
        """
        周期性地把状态刷到磁盘，这样崩溃也不会把游标回退到上次优雅
        关闭的时刻（否则可能重发回复）。
        :param force: 不理会间隔，立即保存
        """
        if self._state_file is None:
            return
        now = time.monotonic()
        if force or (now - self._last_state_save) >= STATE_SAVE_INTERVAL_SEC:
            try:
                self._state.save(self._state_file)
                self._last_state_save = now
            except OSError as e:
                logger.error("Failed to save the inbox state: %s", e)

    def _process_unread_threads(self) -> None:
        """
        如有需要，加载所有未读会话并处理它们。
        """
        counter = -1
        for counter, unread_thread in enumerate(
            self._service.iter_unread_threads()
        ):
            # 更新最后的 history ID，避免重复处理同样的会话
            self._state.update_last_history_id(int(unread_thread.history_id))
            if not unread_thread.messages:
                continue

            # 只对会话中的最后一封邮件发出事件
            self.emit_message_added_event(unread_thread.messages[-1])
            if counter % 100 == 99:
                logger.info("Processed %i unread threads", counter + 1)
            self._save_state_if_due()

        # 记录处理完的未读会话数量
        logger.info("Processed all unread threads (%i)", counter + 1)

        # 更新状态，未读会话不会被再次处理
        self._state.process_all_unread_threads = False
        self._save_state_if_due(force=True)

    def run(self) -> None:
        """
        启动监听器，运行监听 Gmail 收件箱新事件的循环。
        """
        if self._state.process_all_unread_threads:
            self._process_unread_threads()

        while True:
            # 还没有最后的 history ID 时，从 Google Gmail 服务取当前
            # 最大的一个，从这里开始处理
            if self._state.last_history_id is None:
                current_max_history_id = self._service.load_max_history_id()
                assert current_max_history_id is not None
                self._state.update_last_history_id(current_max_history_id)

            # 开始遍历历史前先记录状态
            logger.info("Current state: %s", self._state)

            # 从最后已知的 history ID 起获取历史
            counter = -1
            assert self._state.last_history_id is not None
            try:
                history_generator = self._service.iter_history(
                    self._state.last_history_id
                )
                for counter, history in enumerate(history_generator):
                    # 更新最后的 history ID，避免重复处理同样的历史
                    self._state.update_last_history_id(int(history.id))

                    # 遍历新增邮件并调用处理器
                    for message_added in history.messages_added:
                        self.emit_message_added_event(message_added.message)

                    # 遍历删除邮件并调用处理器
                    for message_deleted in history.messages_deleted:
                        self.emit_message_deleted_event(message_deleted.message)

                    if counter % 100 == 99:
                        logger.info("Processed %i history events", counter + 1)
                    self._save_state_if_due()

                # 记录处理完的历史事件数量
                logger.info("Processed %i history events", counter + 1)
            except HistoryExpiredError as e:
                # 存储的游标已早于 Gmail 的历史保留期（约一周）。回退到
                # 全量重扫未读会话，否则会永远轮询这个过期 ID。
                logger.warning(
                    "%s. Falling back to a full rescan of unread threads.", e
                )
                self._state.last_history_id = None
                self._state.process_all_unread_threads = True
                self._save_state_if_due(force=True)
                continue

            # 等待一个轮询间隔，避免打爆 Gmail API
            time.sleep(self._polling_time_sec)

    def state(self) -> GmailInboxState:
        """
        获取监听器当前的状态。便于保存到磁盘并在重启后加载。
        :return:
        """
        return self._state

    def emit_message_added_event(self, message: models.Message):
        """
        向所有已注册的处理器发出邮件新增事件。
        :param message: 要发出事件的邮件
        """
        logger.debug("Emitting the message added event %s", message)
        event = events.MessageAddedEvent(self._service, message)
        for handler in self._handlers:
            try:
                handler.on_message_added(event)
            except Exception as e:
                # 处理器失败不能静默吞掉消息：大声记录日志，让失败
                # 可见、可以人工重试。
                logger.exception(e)
                logger.error(
                    "Error while handling the message added event %s — the "
                    "message was NOT processed successfully",
                    message.id,
                )

    def emit_message_deleted_event(self, message: models.Message):
        """
        向所有已注册的处理器发出邮件删除事件。
        :param message: 要发出事件的邮件
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
