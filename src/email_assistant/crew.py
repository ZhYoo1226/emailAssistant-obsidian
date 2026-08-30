from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tasks import TaskOutput
from crewai.tasks.conditional_task import ConditionalTask

import config
from email_assistant import models
from email_assistant.storage import QdrantStorage
from email_assistant.tools.qdrant_tool.tool import (
    QdrantHybridSearchTool,
)


def _gateway_llm(model: str) -> LLM:
    # crewai 1.15：Agent 没有 `timeout` 参数（传了也会被静默忽略）；
    # 超时应设置在 LLM 实例上，由它传给 litellm。
    # LLM 无显式 __init__，timeout 是 pydantic 字段（运行时合法），
    # 但基类 BaseLLM 的 stub 未声明它，Pylance 误报——忽略。
    return LLM(  # pyright: ignore[reportCallIssue]
        model=f"openai/{model}", timeout=config.LLM_TIMEOUT_SEC
    )


class BaseCrew:
    """
    项目中各 crew 的基类。
    """

    def __init__(
        self,
        embedder_config: dict,
        qdrant_location: str,
        qdrant_api_key: str | None = None,
        sparse_embedder: Any | None = None,
        reranker: Any | None = None,
    ):
        self.embedder_config = embedder_config
        self.qdrant_location = qdrant_location
        self.qdrant_api_key = qdrant_api_key
        self.sparse_embedder = sparse_embedder
        self.reranker = reranker
        # 单个共享的 storage 实例：搜索工具和 handler 都使用它，
        # 所以不能在每次 knowledge_base() 调用时各建一个
        # （那样会各自打开 Qdrant 客户端并重新加载嵌入模型）。
        self._knowledge_base: QdrantStorage | None = None

    def knowledge_base(self) -> QdrantStorage:
        if self._knowledge_base is None:
            self._knowledge_base = QdrantStorage(
                type="knowledge-base",
                embedder_config=self.embedder_config,
                qdrant_location=self.qdrant_location,
                qdrant_api_key=self.qdrant_api_key,
                sparse_embedder=self.sparse_embedder,
                reranker=self.reranker,
            )
        return self._knowledge_base


@CrewBase
class KnowledgeOrganizingCrew(BaseCrew):
    """
    负责处理原始文本数据、将其转化为结构化知识的 crew。
    """

    agents_config = "config/knowledge/agents.yaml"
    tasks_config = "config/knowledge/tasks.yaml"

    # CrewBase 的元类会在运行时改写这些类属性：config 变成从 YAML 加载
    # 的 dict，agents/tasks 由装饰器注入。stub 把它们标注为 list，所以
    # 下面按字符串取值时需要逐行 ignore。

    @agent
    def chunks_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["chunks_extractor"],  # pyright: ignore[reportArgumentType]
            verbose=True,
            llm=_gateway_llm(config.gateway_fast_model),
        )

    @agent
    def contextualizer(self) -> Agent:
        return Agent(
            config=self.agents_config["contextualizer"],  # pyright: ignore[reportArgumentType]
            verbose=True,
            llm=_gateway_llm(config.gateway_fast_model),
        )

    @task
    def extract_chunks(self) -> Task:
        return Task(  # pyright: ignore[reportCallIssue]
            config=self.tasks_config["extract_chunks"],  # pyright: ignore[reportArgumentType]
            output_pydantic=models.Chunks,
        )

    @task
    def contextualize_chunks(self) -> Task:
        # 任务描述借鉴自 Anthropic 的 Contextual Retrieval
        # 参见：https://www.anthropic.com/news/contextual-retrieval/
        return Task(  # pyright: ignore[reportCallIssue]
            config=self.tasks_config["contextualize_chunks"],  # pyright: ignore[reportArgumentType]
            output_pydantic=models.ContextualizedChunks,
        )

    @crew
    def crew(self) -> Crew:
        """创建 KnowledgeOrganizingCrew"""
        # memory=False，嵌入器永远不会被用到；传原始 dict 也会
        # 触发 crewai 1.15 更严格的 EmbedderConfig 校验失败。
        return Crew(
            agents=self.agents,  # pyright: ignore[reportAttributeAccessIssue]
            tasks=self.tasks,  # pyright: ignore[reportAttributeAccessIssue]
            process=Process.sequential,
            memory=False,
            verbose=True,
        )


@CrewBase
class AutoResponderCrew(BaseCrew):
    """自动回复 crew"""

    agents_config = "config/autoresponder/agents.yaml"
    tasks_config = "config/autoresponder/tasks.yaml"

    @agent
    def categorizer(self) -> Agent:
        return Agent(
            config=self.agents_config["categorizer"],  # pyright: ignore[reportArgumentType]
            verbose=True,
            llm=_gateway_llm(config.gateway_fast_model),
        )

    @agent
    def response_writer(self) -> Agent:
        agent_kwargs = {}
        if config.response_temperature is not None:
            agent_kwargs["temperature"] = config.response_temperature
        return Agent(
            config=self.agents_config["response_writer"],  # pyright: ignore[reportArgumentType]
            tools=[
                QdrantHybridSearchTool(self.knowledge_base()),
            ],
            verbose=True,
            llm=_gateway_llm(config.gateway_main_model),
            max_iter=4,
            **agent_kwargs,
        )

    @task
    def categorization_task(self) -> Task:
        return Task(  # pyright: ignore[reportCallIssue]
            config=self.tasks_config["categorization_task"],  # pyright: ignore[reportArgumentType]
            output_pydantic=models.EmailThreadCategories,
        )

    @task
    def response_writing_task(self):
        return ConditionalTask(  # pyright: ignore[reportCallIssue]
            config=self.tasks_config["response_writing_task"],  # pyright: ignore[reportArgumentType]
            output_pydantic=models.EmailResponse,
            condition=self.is_a_question,
        )

    @crew
    def crew(self) -> Crew:
        """创建 AutoResponderCrew"""
        # memory=False，嵌入器永远不会被用到；传原始 dict 也会
        # 触发 crewai 1.15 更严格的 EmbedderConfig 校验失败。
        return Crew(
            agents=self.agents,  # pyright: ignore[reportAttributeAccessIssue]
            tasks=self.tasks,  # pyright: ignore[reportAttributeAccessIssue]
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    def is_a_question(self, output: TaskOutput) -> bool:
        # TaskOutput.pydantic 的类型标注是 BaseModel | None；运行时它
        # 持有上一个任务的 output_pydantic 模型。
        email_thread_categories = output.pydantic  # pyright: ignore[reportAssignmentType]
        assert isinstance(email_thread_categories, models.EmailThreadCategories)
        return "QUESTION" in email_thread_categories.categories
