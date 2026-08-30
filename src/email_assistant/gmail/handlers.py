import abc
import logging
import os
import re

from litellm import completion
from markdownify import markdownify

import config
from email_assistant import models
from email_assistant.crew import AutoResponderCrew
from email_assistant.gmail import events

logger = logging.getLogger(__name__)

# 回复任务指示模型在知识库给不出答案时写下的哨兵文本（见
# config/autoresponder/tasks.yaml）。它是给程序看的，不是给收件人看的，
# 绝不能原样发送出去。
NO_RESPONSE_SENTINEL = "I cannot provide a response"

# 模型可能把引用标记泄漏进回复文本（如 "blodguy_ink[来源1]"），
# 尽管提示词已禁止。发送前先剥掉。
_CITATION_MARKER_RE = re.compile(
    r"\s*[\[\(](?:来源|source|ref(?:erence)?)\s*\d+[\]\)]", re.IGNORECASE
)
_BARE_INDEX_RE = re.compile(r"\s*\[\d+\]")


class GmailInboxEventHandler(abc.ABC):  # noqa: B024 — optional-handler pattern
    """
    Gmail 收件箱中所有事件的通用处理器。
    """

    def on_message_added(self, event: events.MessageAddedEvent):  # noqa: B027
        """
        处理新邮件到达 Gmail 收件箱的事件。
        :param event: 要处理的事件
        """
        pass

    def on_message_deleted(self, event: events.MessageDeletedEvent):  # noqa: B027
        """
        处理邮件从 Gmail 收件箱中删除的事件。
        :param event: 要处理的事件
        """
        pass


class AgenticAutoReplyHandler(GmailInboxEventHandler):
    """
    给邮件发件人发送自动回复的事件处理器。
    """

    def __init__(
        self,
        embedder_config: dict,
        qdrant_location: str,
        qdrant_api_key: str | None = None,
        sparse_embedder=None,
        reranker=None,
    ):
        # CrewBase 的 TYPE_CHECKING stub 向 pyright 隐藏了 BaseCrew 的
        # 真实签名（详见 obsidian/handlers.py）。
        crew_builder = AutoResponderCrew(
            embedder_config,  # pyright: ignore[reportCallIssue]
            qdrant_location,
            qdrant_api_key,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
        )
        self.crew = crew_builder.crew()
        self.knowledge_base = crew_builder.knowledge_base()  # pyright: ignore[reportAttributeAccessIssue]

    def on_message_added(self, event: events.MessageAddedEvent):
        """
        处理新邮件到达 Gmail 收件箱的事件。
        :param event: 要处理的事件
        """
        service = event.service()
        message = event.message()
        logger.info("Received a new message: %s", message)

        # 加载完整会话
        thread = service.load_full_thread(message.thread_id)
        last_message = thread.messages[-1]
        last_labels = last_message.label_ids or []

        # 只处理未读邮件，已读的直接跳过
        if "UNREAD" not in last_labels:
            logger.info("The last message is already read. Skipping the reply.")
            return

        # 最后一封邮件若是草稿，则不回复
        if "DRAFT" in last_labels:
            logger.info("The last message is already a draft. Skipping the reply.")
            return

        # 解码各封邮件并转成 Markdown。跳过草稿——它们是系统自己之前
        # 的回复，不能喂回给分类器（否则会话会被错误地归为垃圾邮件）。
        non_draft_messages = [
            message for message in thread.messages
            if "DRAFT" not in (message.label_ids or [])
        ]
        decoded_messages = [
            service.decode_message(message) for message in non_draft_messages
        ]
        md_messages = [
            markdownify(decoded_message.content) for decoded_message in decoded_messages
        ]

        # 调用 crew 生成回复
        response = self.crew.kickoff(inputs={"messages": md_messages})
        pydantic_output = getattr(response, "pydantic", None)
        logger.info("Generated response: %s", pydantic_output)
        if not isinstance(pydantic_output, models.EmailResponse):
            logger.info(
                "Crew decided not to respond to the message: %s", pydantic_output
            )
            return

        # 用生成的回复创建草稿
        email_response = pydantic_output
        if email_response.content is None:
            logger.info("The response is empty. Skipping the reply.")
            return

        # 发送前校验引用：被引用的来源必须是知识库中真实存在的文件，
        # 且回复必须忠实于来源。校验不通过时发送得体的“超出范围”回复，
        # 而不是保持沉默。
        if not self._verify_sources(email_response):
            logger.warning("Sources could not be verified. Sending fallback reply.")
            service.send_message(
                thread, content=self._fallback_reply(last_message.snippet)
            )
            return
        if not self._verify_faithfulness(email_response):
            logger.warning("Response not faithful to sources. Sending fallback reply.")
            service.send_message(
                thread, content=self._fallback_reply(last_message.snippet)
            )
            return

        if NO_RESPONSE_SENTINEL in (email_response.content or ""):
            logger.warning(
                "Response is the no-answer sentinel. Sending fallback reply."
            )
            service.send_message(
                thread, content=self._fallback_reply(last_message.snippet)
            )
            return

        service.send_message(
            thread,
            content=self._strip_citation_markers(email_response.content) or "",
        )

    @staticmethod
    def _strip_citation_markers(content: str | None) -> str | None:
        if not content:
            return content
        stripped = _CITATION_MARKER_RE.sub("", content)
        stripped = _BARE_INDEX_RE.sub("", stripped)
        return stripped.strip() or content

    def _verify_sources(self, email_response: models.EmailResponse) -> bool:
        """
        检查每个被引用的 src_path 是否真实存在于知识库中。
        空的来源列表是允许的（如 "I cannot provide a response" 的情形）。
        """
        if not email_response.sources:
            return True

        def _normalize(path: str) -> str:
            # 模型常输出 JSON 转义或 posix 风格的路径；在归一化形式上
            # 比较，避免真实的引用被误拒。
            return os.path.normcase(path.replace("\\\\", "\\").replace("/", "\\"))

        indexed_paths = {
            _normalize(p) for p in self.knowledge_base.list_src_paths()
        }
        for source in email_response.sources:
            if _normalize(source.src_path) not in indexed_paths:
                logger.warning(
                    "Cited source not in knowledge base: %s", source.src_path
                )
                return False
        return True

    def _verify_faithfulness(self, email_response: models.EmailResponse) -> bool:
        """
        用快速模型判断回复中的事实性陈述是否被引用来源支持
        （用低成本的 LLM 代替专门的 NLI 模型）。
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
        # litellm 的 stub 把同步返回类型标注成了 CustomStreamWrapper；
        # 实际的非流式响应带有 .choices。
        response = completion(
            model=f"openai/{config.gateway_fast_model}",
            messages=[{"role": "user", "content": prompt}],
            timeout=config.LLM_TIMEOUT_SEC,
        )
        verdict = (response.choices[0].message.content or "").strip().upper()  # pyright: ignore[reportAttributeAccessIssue]
        logger.info("Faithfulness verdict: %s", verdict)
        return verdict.startswith("YES")

    def _fallback_reply(self, question_snippet: str | None) -> str:
        """
        知识库无法支撑一个忠实回答时发送的得体的“超出范围”回复。
        让发件人知道正在回复的是 AI 助手，问题需要本人关注。
        """
        question = (question_snippet or "您的问题").strip()[:10]
        return (
            "<p>你好！我是这边的邮件小助手。</p>"
            f"<p>你问的「{question}」我暂时拿不准，就不瞎答了，怕误导你。</p>"
            "<p>本人看到邮件后会尽快回复你；如果比较急，也可以先通过其他方式联系～</p>"
        )
