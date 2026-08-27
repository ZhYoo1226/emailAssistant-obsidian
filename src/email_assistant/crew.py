import abc
import os
from typing import Optional

from crewai import Agent, Crew, Process, Task
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


class BaseCrew(abc.ABC):
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

    def knowledge_base(self) -> QdrantStorage:
        return QdrantStorage(
            type="knowledge-base",
            embedder_config=self.embedder_config,
            qdrant_location=self.qdrant_location,
            qdrant_api_key=self.qdrant_api_key,
        )


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
            llm=f"openai/{GATEWAY_FAST_MODEL}",
        )

    @agent
    def contextualizer(self) -> Agent:
        return Agent(
            config=self.agents_config["contextualizer"],
            verbose=True,
            llm=f"openai/{GATEWAY_FAST_MODEL}",
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
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            memory=False,
            embedder=self.embedder_config,
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
            llm=f"openai/{GATEWAY_FAST_MODEL}",
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
            llm=f"openai/{GATEWAY_MAIN_MODEL}",
            max_iter=6,
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
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            memory=False,
            embedder=self.embedder_config,
            verbose=True,
        )

    def is_a_question(self, output: TaskOutput) -> bool:
        email_thread_categories: models.EmailThreadCategories = output.pydantic  # noqa
        return "QUESTION" in email_thread_categories.categories
