import os
from typing import Optional

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tasks import TaskOutput
from crewai.tasks.conditional_task import ConditionalTask

from email_assistant import models
from email_assistant.tools.qdrant_tool.tool import (
    QdrantSearchTool,
)
from email_assistant.storage import QdrantStorage

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
        qdrant_api_key: Optional[str] = None,
    ):
        self.embedder_config = embedder_config
        self.qdrant_location = qdrant_location
        self.qdrant_api_key = qdrant_api_key
        # A single shared storage instance: both the search tool and the
        # handler use it, so we must not create one per knowledge_base() call
        # (each would open its own Qdrant client and reload the embedder).
        self._knowledge_base: Optional[QdrantStorage] = None

    def knowledge_base(self) -> QdrantStorage:
        if self._knowledge_base is None:
            self._knowledge_base = QdrantStorage(
                type="knowledge-base",
                embedder_config=self.embedder_config,
                qdrant_location=self.qdrant_location,
                qdrant_api_key=self.qdrant_api_key,
            )
        return self._knowledge_base


@CrewBase
class KnowledgeOrganizingCrew(BaseCrew):
    """
    A crew responsible for processing raw text data and converting it into structured knowledge.
    """

    agents_config = "config/knowledge/agents.yaml"
    tasks_config = "config/knowledge/tasks.yaml"

    @agent
    def chunks_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["chunks_extractor"],
            verbose=True,
            llm=_gateway_llm(GATEWAY_FAST_MODEL),
        )

    @agent
    def contextualizer(self) -> Agent:
        return Agent(
            config=self.agents_config["contextualizer"],
            verbose=True,
            llm=_gateway_llm(GATEWAY_FAST_MODEL),
        )

    @task
    def extract_chunks(self) -> Task:
        return Task(
            config=self.tasks_config["extract_chunks"],
            output_pydantic=models.Chunks,
        )

    @task
    def contextualize_chunks(self) -> Task:
        # The task description is borrowed from the Anthropic Contextual Retrieval
        # See: https://www.anthropic.com/news/contextual-retrieval/
        return Task(
            config=self.tasks_config["contextualize_chunks"],
            output_pydantic=models.ContextualizedChunks,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the KnowledgeOrganizingCrew crew"""
        # memory=False, so the embedder would never be used; passing the raw
        # dict also fails crewai 1.15's stricter EmbedderConfig validation.
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
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
            config=self.agents_config["categorizer"],
            verbose=True,
            llm=_gateway_llm(GATEWAY_FAST_MODEL),
        )

    @agent
    def response_writer(self) -> Agent:
        agent_kwargs = {}
        if RESPONSE_TEMPERATURE is not None:
            agent_kwargs["temperature"] = RESPONSE_TEMPERATURE
        return Agent(
            config=self.agents_config["response_writer"],
            tools=[
                QdrantSearchTool(self.knowledge_base()),
            ],
            verbose=True,
            llm=_gateway_llm(GATEWAY_MAIN_MODEL),
            max_iter=4,
            **agent_kwargs,
        )

    @task
    def categorization_task(self) -> Task:
        return Task(
            config=self.tasks_config["categorization_task"],
            output_pydantic=models.EmailThreadCategories,
        )

    @task
    def response_writing_task(self):
        return ConditionalTask(
            config=self.tasks_config["response_writing_task"],
            output_pydantic=models.EmailResponse,
            condition=self.is_a_question,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AutoResponderCrew crew"""
        # memory=False, so the embedder would never be used; passing the raw
        # dict also fails crewai 1.15's stricter EmbedderConfig validation.
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

    def is_a_question(self, output: TaskOutput) -> bool:
        email_thread_categories: models.EmailThreadCategories = output.pydantic  # noqa
        return "QUESTION" in email_thread_categories.categories
