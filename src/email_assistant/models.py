from pydantic import BaseModel, Field


class EmailThreadCategories(BaseModel):
    categories: list[str]


class EmailSource(BaseModel):
    index: int = Field(
        description="The citation number used internally to match the source with the response"
    )
    src_path: str = Field(description="The source file path")
    snippet: str = Field(description="The retrieved snippet that supports the response")


class EmailResponse(BaseModel):
    content: str = Field(
        description="HTML content of the email response, without any citation markers"
    )
    sources: list[EmailSource] = Field(
        description="The sources the response relied on, for internal verification only"
    )


class Chunk(BaseModel):
    content: str = Field(description="The content of the chunk")


class Chunks(BaseModel):
    chunks: list[Chunk] = Field(
        description="A list of chunks extracted from the document"
    )


class ContextualizedChunk(Chunk):
    context: str = Field(
        description="The context of the chunk in relation to the document"
    )


class ContextualizedChunks(BaseModel):
    chunks: list[ContextualizedChunk] = Field(
        description="A list of contextualized chunks extracted from the document"
    )
