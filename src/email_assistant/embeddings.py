"""
检索服务的实现者：基于 fastembed ONNX 在本地运行的稠密嵌入、
稀疏（BM25）嵌入和交叉编码器重排器。只定义实现，不读配置——
模型名由调用方从 config 传入。
"""

import jieba
from chromadb import EmbeddingFunction
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder


class FastEmbedFunction(EmbeddingFunction):
    """基于本地 fastembed 模型、兼容 chromadb 的 EmbeddingFunction。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def name(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"fastembed/{self.model_name}"

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return [vector.tolist() for vector in self._model.embed(input)]


class JiebaBM25Function:
    """
    面向中文文本的稀疏嵌入：先用 jieba 把文本切分成空格分隔的词，
    再交给 fastembed 的 BM25 转成稀疏向量。如果不预分词，BM25 会退回
    按空白切分，整句中文会被压成 1-2 个 token。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name)

    @staticmethod
    def _segment(texts: list[str]) -> list[str]:
        return [" ".join(jieba.cut_for_search(text)) for text in texts]

    def embed(self, texts: list[str]):
        return list(self._model.embed(self._segment(texts)))


class BgeRerankFunction:
    """对 (查询, 候选) 对打分的本地交叉编码器重排器。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(score) for score in self._model.rerank(query, documents)]
