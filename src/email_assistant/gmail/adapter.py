import base64
import logging
import time
from collections.abc import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import google_auth_httplib2
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

import config
from email_assistant.gmail import models

logger = logging.getLogger(__name__)


class HistoryExpiredError(Exception):
    """
    当 Gmail 不再保留所请求的 startHistoryId 的历史记录时抛出（HTTP 404）。
    调用方必须回退到全量重扫，否则会永远轮询这个过期 ID 并静默地什么都不做。
    """


class GmailServiceAdapter:
    """
    Gmail API 服务的外层适配器：简化与 API 的交互，用结构化的数据类
    取代原始 JSON 对象。
    """

    CREDENTIALS_FILE_NAME = "credentials.json"
    TOKEN_FILE_NAME = "token.json"
    GOOGLE_API_SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.send",
    ]
    DEFAULT_CHARSET = "utf-8"
    CONTENT_TYPE_PREFERRED = ["text/html", "text/plain"]
    _CONNECTION_RETRY_DELAY_SEC = 30

    def __init__(self, credentials_dir: Path):
        self._credentials_dir = credentials_dir
        if not self._credentials_dir.exists():
            self._credentials_dir.mkdir(parents=True)
        self._credentials: Credentials | None = None
        self._service: Resource | None = None

    def is_authenticated(self) -> bool:
        """
        检查用户是否已通过 Google API 认证。
        """
        return self._service is not None

    def authenticate(self):
        """
        用 Gmail API 认证用户。如果用户尚未授权本应用，会重定向到
        Google 登录页面进行授权。
        """
        token_file = self._credentials_dir / self.TOKEN_FILE_NAME
        if token_file.exists():
            self._credentials = Credentials.from_authorized_user_file(
                str(token_file), self.GOOGLE_API_SCOPES
            )

        # 没有有效凭据，用户需要登录
        credentials_file = self._credentials_dir / self.CREDENTIALS_FILE_NAME
        if (
            self._credentials is not None
            and self._credentials.expired
            and self._credentials.refresh_token
        ):
            # 访问令牌已过期但还有刷新令牌——静默刷新，
            # 而不是让用户重新授权。
            self._credentials.refresh(Request())
        elif self._credentials is None or not self._credentials.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_file), self.GOOGLE_API_SCOPES
            )
            # stub 把 run_local_server 的返回类型标注成了与
            # external_account_authorized_user.Credentials 的联合类型，
            # 但 InstalledAppFlow 返回的一定是 oauth2 的 Credentials。
            self._credentials = flow.run_local_server(port=0)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self._credentials.refresh(Request())

        assert self._credentials is not None
        # 保存凭据供下次运行使用
        with open(token_file, "w") as fp:
            fp.write(self._credentials.to_json())

        # 连接 Gmail 服务。httplib2 会忽略 HTTP(S)_PROXY 环境变量，
        # 所以配置了代理时要显式构建带代理的 http 对象，否则在防火墙
        # 网络下对 googleapis.com 的请求会超时。
        import httplib2

        proxy_url = config.https_proxy
        if proxy_url:
            from urllib.parse import urlparse

            parsed = urlparse(proxy_url if "//" in proxy_url else f"http://{proxy_url}")
            pi = httplib2.ProxyInfo(
                proxy_type=httplib2.socks.PROXY_TYPE_HTTP,  # pyright: ignore[reportAttributeAccessIssue]
                proxy_host=parsed.hostname,
                proxy_port=parsed.port or 8080,
            )
            http = httplib2.Http(proxy_info=pi)
            authorized_http = google_auth_httplib2.AuthorizedHttp(
                self._credentials, http=http
            )
            self._service = build(
                "gmail", "v1", http=authorized_http, cache_discovery=False
            )
        else:
            self._service = build(
                "gmail", "v1", credentials=self._credentials, cache_discovery=False
            )

    def _gmail(self) -> Any:
        """
        Gmail 的 users() 资源。googleapiclient 的 Resource 是动态生成的
        ——stub 里不存在 .users()——所以调用方拿到的是 Any。
        """
        assert self._service is not None
        return self._service.users()  # pyright: ignore[reportAttributeAccessIssue]

    def _execute_with_retry(self, request):
        """
        执行一次 Gmail API 请求；遇到瞬时连接错误（本地代理断开、
        WinError 10053 等）时退避重试，而不是让异常杀死监听线程。
        :param request: 执行 API 调用的无参 callable
        :return: 解析后的响应
        """
        while True:
            try:
                return request()
            except HttpError:
                raise
            except (ConnectionError, OSError, TimeoutError) as e:
                logger.error(
                    "Connection error occurred: %s. Retrying in %ss...",
                    e, self._CONNECTION_RETRY_DELAY_SEC,
                )
                time.sleep(self._CONNECTION_RETRY_DELAY_SEC)

    def iter_unread_threads(self) -> Generator[models.Thread, None, None]:
        """
        遍历 Gmail 收件箱中所有未读会话。
        :return: 未读会话的生成器
        """
        # 分页遍历所有未读会话
        page_token = None
        while True:
            def _fetch_page(token=page_token):
                return (
                    self._gmail()
                    .threads()
                    .list(userId="me", q="is:unread", pageToken=token)
                    .execute()
                )

            response = self._execute_with_retry(_fetch_page)
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
        从给定的 history ID 开始遍历收件箱的历史记录。
        :param last_history_id: 起始的 history ID
        :return: 历史记录对象的生成器
        """
        # 分页遍历所有历史记录
        page_token = None
        while True:
            try:
                response = (
                    self._gmail()
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
                # 404 表示 Gmail 不再保留所请求的 startHistoryId 的历史
                # （历史大约一周后过期）。抛出异常让调用方回退到全量
                # 重扫，而不是永远静默轮询一个过期游标。
                if e.resp.status == 404:
                    raise HistoryExpiredError(
                        f"History for startHistoryId={last_history_id} has expired"
                    ) from e
                logger.error("HTTP error occurred: %s", e)
                return
            except (ConnectionError, OSError) as e:
                # 本地代理或网络可能在请求中途断开已建立的连接
                # （如 WinError 10053）。退避后重试，而不是让异常
                # 永久杀死轮询线程。
                logger.error(
                    "Connection error occurred: %s. Retrying in %ss...",
                    e,
                    self._CONNECTION_RETRY_DELAY_SEC,
                )
                time.sleep(self._CONNECTION_RETRY_DELAY_SEC)
                continue

    def load_max_history_id(self) -> int | None:
        """
        从 Gmail 服务加载最大 history ID。只取最后一封邮件并读取其
        history ID。
        :return: 最大 history ID
        """
        messages = self._execute_with_retry(
            lambda: (
                self._gmail()
                .messages()
                .list(userId="me", maxResults=1)
                .execute()
            )
        )
        if not messages.get("messages", []):
            return None
        message_id = messages["messages"][0]["id"]
        last_message = self._execute_with_retry(
            lambda: (
                self._gmail()
                .messages()
                .get(userId="me", id=message_id)
                .execute()
            )
        )
        return int(last_message["historyId"])

    def load_full_thread(self, thread_id: str) -> models.Thread:
        """
        从 Gmail 服务加载完整会话。
        :param thread_id: 要加载的会话 ID
        :return: 完整的会话对象
        """
        full_thread = self._execute_with_retry(
            lambda: (
                self._gmail()
                .threads()
                .get(userId="me", id=thread_id, format="full")
                .execute()
            )
        )
        return models.Thread(**full_thread)

    def load_full_message(self, message_id: str) -> models.Message:  # noqa: B019
        """
        从 Gmail 服务加载完整邮件。
        :param message_id: 要加载的邮件 ID
        :return: 完整的邮件对象
        """
        full_message = self._execute_with_retry(
            lambda: (
                self._gmail()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        )
        return models.Message(**full_message)

    def decode_message(self, message: models.Message) -> models.DecodedMessage:
        """
        从 base64 编码中解码邮件内容。
        """
        content = self._extract_message_content(message)
        return models.DecodedMessage(message=message, content=content)

    def send_message(self, thread: models.Thread, content: str):
        """
        向会话发送回复。接收邮件的 HTML 内容，内部转成纯文本，
        然后直接发送（不是存草稿）。
        """
        # 用最后一封邮件获取所有元数据
        last_message = thread.messages[-1]

        # 幂等保护：如果会话中已存在对同一封邮件（通过 In-Reply-To 匹配）
        # 的回复，再发一次就会重复。这覆盖 handler 的重复投递和发送中途
        # 断连后的重试场景。
        reply_to_id = last_message.get_header_value("Message-ID")
        if reply_to_id and self._thread_already_answered(thread, reply_to_id):
            logger.info(
                "Thread already has a reply to message %s. Skipping send.",
                reply_to_id,
            )
            return

        # 个别特殊邮件可能缺少头字段；没有 From/To 就无法构建有效的
        # 回复，跳过而不是崩溃。
        to_value = last_message.get_header_value("From")
        from_value = last_message.get_header_value("To")
        if not to_value or not from_value:
            logger.warning(
                "Missing From/To headers, cannot build a reply: %s", last_message.id
            )
            return
        subject = last_message.get_header_value("Subject") or ""

        # 去掉 HTML 生成纯文本版本
        plain_content = BeautifulSoup(content, "html.parser").get_text()

        # 构建邮件消息
        email_message = MIMEMultipart("alternative")
        email_message["To"] = self._parse_email(to_value)
        email_message["From"] = self._parse_email(from_value)
        email_message["Subject"] = subject
        if reply_to_id:
            email_message["In-Reply-To"] = reply_to_id
            email_message["References"] = reply_to_id

        # 创建纯文本和 HTML 两个部分
        plain_part = MIMEText(plain_content, "plain")
        html_part = MIMEText(content, "html")

        # 把两部分挂到邮件上
        email_message.attach(plain_part)
        email_message.attach(html_part)

        # 直接发送。显式携带 threadId 强制归并进原会话——仅靠
        # In-Reply-To/References 头 Gmail 并不总能归并（对 QQ 邮箱发来的
        # 邮件实测会散成新线程），而归并失效会让幂等检查（在原线程里找
        # 回复）永远查不到已发过的回复。连接中断（如代理抖动）会中途
        # 放弃发送；带退避地重试几次，避免瞬时网络抖动静默丢失回复。
        raw = base64.urlsafe_b64encode(
            email_message.as_string().encode("utf-8")
        ).decode()

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                sent = (
                    self._gmail()
                    .messages()
                    .send(
                        userId="me",
                        body={"raw": raw, "threadId": thread.id},
                    )
                    .execute()
                )
                logger.info("Sent a reply message: %s", sent)
                return
            except (ConnectionError, OSError) as e:
                last_error = e
                # 上一次尝试可能在断连之前已经到达 Gmail。重新加载
                # 会话并检查回复是否已送达，再决定是否重试，
                # 避免重复发送。
                try:
                    fresh_thread = self.load_full_thread(thread.id)
                    if reply_to_id and self._thread_already_answered(
                        fresh_thread, reply_to_id
                    ):
                        logger.info(
                            "Reply to message %s was delivered by a previous "
                            "attempt. Skipping retry.",
                            reply_to_id,
                        )
                        return
                except HttpError:
                    pass
                delay = 10 * (attempt + 1)
                logger.warning(
                    "Send attempt %i/3 failed: %s. Retrying in %is...",
                    attempt + 1, e, delay,
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def mark_message_read(self, message_id: str) -> None:
        """
        把邮件标记为已读（移除 UNREAD 标签）。回复发出后调用，否则
        来信在 Gmail 眼里永远未读，任何一次全量重扫未读会话都会把
        旧邮件再翻出来回复一遍。
        :param message_id: 要标记的邮件 ID
        """
        self._execute_with_retry(
            lambda: (
                self._gmail()
                .messages()
                .modify(
                    userId="me",
                    id=message_id,
                    body={"removeLabelIds": ["UNREAD"]},
                )
                .execute()
            )
        )

    def _thread_already_answered(self, thread: models.Thread, message_id: str) -> bool:
        """
        检查会话中是否已存在针对给定 Message-ID 的回复
        （即某封邮件的 In-Reply-To 引用了它）。
        :param thread: 要检查的会话
        :param message_id: 被回复邮件的 Message-ID
        :return: 已存在回复则返回 True
        """
        for message in thread.messages:
            in_reply_to = message.get_header_value("In-Reply-To")
            if in_reply_to and message_id in in_reply_to:
                return True
        return False

    def _extract_message_content(self, message: models.Message) -> str:
        """
        从邮件对象中提取正文内容。
        """
        payload = message.payload
        if payload is None:
            raise ValueError("Message has no payload.")

        content: str | None = None
        charset: str = self.DEFAULT_CHARSET
        for mime_type in self.CONTENT_TYPE_PREFERRED:
            # 统一转成小写再比较
            mime_type = mime_type.lower()

            # body 里如果有数据就直接用
            if (
                payload.mime_type.lower() == mime_type
                and payload.body.data
            ):
                content = payload.body.data
                charset = self._extract_content_charset(payload)
                logger.debug(f"Found {mime_type} body with charset {charset}")
                break

            # 有些邮件没有 parts，所以要先确认 parts 是否存在
            if not payload.parts:
                continue

            # body 没有数据时检查各部分，先把嵌套的 parts 拍平成一个列表
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

            # 找到内容就跳出循环
            if content:
                break

        # 没有找到文本正文则报错
        if content is None:
            raise ValueError("No text body found.")

        # 内容是 base64 编码的，先解码
        base64_decoded = base64.urlsafe_b64decode(content)
        try:
            return base64_decoded.decode(charset)
        except LookupError:
            # 找不到该字符集时退回默认值
            return base64_decoded.decode(self.DEFAULT_CHARSET)

    def _load_full_history(self, history_descriptor: dict) -> models.History:
        """
        解析 history 描述符并加载完整的 history 对象。
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
        把邮件的嵌套 parts 拍平成单个列表。
        """
        if message.payload is None:
            return []
        parts = [message.payload]
        for part in message.payload.parts or []:
            parts.extend(self._flatten_parts(part))
        return parts

    def _flatten_parts(self, part: models.MessagePart) -> list[models.MessagePart]:
        """
        把单个 part 内部嵌套的 parts 拍平成单个列表。
        :param part:
        :return:
        """
        parts = [part]
        for inner_part in part.parts or []:
            parts.extend(self._flatten_parts(inner_part))
        return parts

    def _extract_content_charset(self, part: models.MessagePart) -> str:
        """
        从 Content-Type 头中提取字符集。
        :param part:
        :return:
        """
        content_type_header = next(
            (
                header
                for header in part.headers
                if header.name.lower() == "content-type"
            ),
            None,
        )
        if content_type_header:
            content_type = content_type_header.value
            charset_index = content_type.find("charset=")
            if charset_index == -1:
                return self.DEFAULT_CHARSET
            charset_index_end = content_type.find(";", charset_index)
            if charset_index_end == -1:
                charset_index_end = len(content_type)
            charset = content_type[charset_index + len("charset=") : charset_index_end]
            return charset.strip().strip('"').strip("'")
        return self.DEFAULT_CHARSET

    def _parse_email(self, text: str) -> str:
        """
        从形如 '"John Done" <johndone@email.com>' 的格式化文本中解析出
        邮件地址，只返回邮箱部分。如果传入的已经是邮件地址，则原样返回。
        :param text:
        :return:
        """
        if "<" in text and ">" in text:
            return text.split("<")[1].split(">")[0]
        return text
