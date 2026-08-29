import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tasks import TaskOutput
from crewai.tasks.conditional_task import ConditionalTask

from email_assistant import models
from email_assistant.storage import QdrantStorage
from email_assistant.tools.qdrant_tool.tool import (
    QdrantHybridSearchTool,
)

# Models served by the OpenAI-compatible gateway (see config.py).
GATEWAY_MAIN_MODEL = os.environ.get("GATEWAY_MAIN_MODEL", "deepseek-v4-pro")
GATEWAY_FAST_MODEL = os.environ.get("GATEWAY_FAST_MODEL", "deepseek-v4-flash")

# Optional temperature override for the response writer (for debugging
# hallucination). Leave unset to use the model default.
RESPONSE_TEMPERATURE = None
if os.environ.get("RESPONSE_TEMPERATURE"):
    RESPONSE_TEMPERATURE = float(os.environ["RESPONSE_TEMPERATURE"])

# Timeout (seconds) for every LLM call, so a hung gateway connection cannot
# block an email-processing or ingestion run forever. DeepSeek thinking-mode
# models can legitimately take a while, so this is generous.
LLM_TIMEOUT_SEC = int(os.environ.get("LLM_TIMEOUT_SEC", "180"))


def _gateway_llm(model: str) -> LLM:
    # crewai 1.15: Agent has no `timeout` param (it was silently ignored);
    # the timeout belongs on the LLM instance, which passes it to litellm.
    return LLM(model=f"openai/{model}", timeout=LLM_TIMEOUT_SEC)


class BaseCrew:
    """
    Base class for the crews in the project.
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
        # A single shared storage instance: both the search tool and the
        # handler use it, so we must not create one per knowledge_base() call
        # (each would open its own Qdrant client and reload the embedder).
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
    A crew responsible for processing raw text data and converting it into structured knowledge.
    """

    agents_config = "config/knowledge/agents.yaml"
    tasks_config = "config/knowledge/tasks.yaml"

    # CrewBase's metaclass rewrites these class attrs at runtime: the configs
    # become dicts loaded from YAML and agents/tasks are injected by the
    # decorators. The stubs type them as lists, so string lookups below need
    # per-line ignores.

    @agent
    def chunks_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["chunks_extractor"],  # pyright: ignore[reportArgumentType]
            verbose=True,
            llm=_gateway_llm(GATEWAY_FAST_MODEL),
        )

    @agent
    def contextualizer(self) -> Agent:
        return Agent(
            config=self.agents_config["contextualizer"],  # pyright: ignore[reportArgumentType]
            verbose=True,
            llm=_gateway_llm(GATEWAY_FAST_MODEL),
        )

    @task
    def extract_chunks(self) -> Task:
        return Task(  # pyright: ignore[reportCallIssue]
            config=self.tasks_config["extract_chunks"],  # pyright: ignore[reportArgumentType]
            output_pydantic=models.Chunks,
        )

    @task
    def contextualize_chunks(self) -> Task:
        # The task description is borrowed from the Anthropic Contextual Retrieval
        # See: https://www.anthropic.com/news/contextual-retrieval/
        return Task(  # pyright: ignore[reportCallIssue]
            config=self.tasks_config["contextualize_chunks"],  # pyright: ignore[reportArgumentType]
            output_pydantic=models.ContextualizedChunks,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the KnowledgeOrganizingCrew crew"""
        # memory=False, so the embedder would never be used; passing the raw
        # dict also fails crewai 1.15's stricter EmbedderConfig validation.
        return Crew(
            agents=self.agents,  # pyright: ignore[reportAttributeAccessIssue]
            tasks=self.tasks,  # pyright: ignore[reportAttributeAccessIssue]
            process=Process.sequential,
            memory=False,
            verbose=True,
        )


@CrewBase
class AutoResponderCrew(BaseCrew):
    """AutoResponderCrew crew"""

    agents_config = "config/autoresponder/agents.yaml"
    tasks_config = "config/autoresponder/tasks.yaml"

    @agent
    def categorizer(self) -> Agent:
        return Agent(
            config=self.agents_config["categorizer"],  # pyright: ignore[reportArgumentType]
            verbose=True,
            llm=_gateway_llm(GATEWAY_FAST_MODEL),
        )

    @agent
    def response_writer(self) -> Agent:
        agent_kwargs = {}
        if RESPONSE_TEMPERATURE is not None:
            agent_kwargs["temperature"] = RESPONSE_TEMPERATURE
        return Agent(
            config=self.agents_config["response_writer"],  # pyright: ignore[reportArgumentType]
            tools=[
                QdrantHybridSearchTool(self.knowledge_base()),
            ],
            verbose=True,
            llm=_gateway_llm(GATEWAY_MAIN_MODEL),
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
        """Creates the AutoResponderCrew crew"""
        # memory=False, so the embedder would never be used; passing the raw
        # dict also fails crewai 1.15's stricter EmbedderConfig validation.
        return Crew(
            agents=self.agents,  # pyright: ignore[reportAttributeAccessIssue]
            tasks=self.tasks,  # pyright: ignore[reportAttributeAccessIssue]
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    def is_a_question(self, output: TaskOutput) -> bool:
        # TaskOutput.pydantic is typed as BaseModel | None; at runtime it holds
        # the previous task's output_pydantic model.
        email_thread_categories = output.pydantic  # pyright: ignore[reportAssignmentType]
        assert isinstance(email_thread_categories, models.EmailThreadCategories)
        return "QUESTION" in email_thread_categories.categories
