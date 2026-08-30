import uuid
from typing import Any

from qdrant_client import QdrantClient, models

# 集合中的命名向量：dense = bge-small 语义向量，
# sparse = BM25 词法向量（jieba 预分词）。混合检索用 RRF 融合两者，
# 再由调用方对融合后的候选结果重排。
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# RRF 融合调参，在 QdrantStorage.query_points 中生效。k 是 RRF 论文中
# 的排名衰减常数（k 越小越偏向头部排名）；weights 平衡两条检索路径
# ——顺序为 [dense, sparse]。
RRF_K = 2
RRF_WEIGHTS = [1.0, 1.0]


class QdrantStorage:
    """
    Qdrant 中知识库条目的存储，用项目的本地 fastembed 模型做嵌入。
    独立类：crewai>=1.15 不再自带 crewai.memory.storage.rag_storage。
    """

    TEST_STRING = "test"
    MAX_LENGTH_BYTES = 8192

    def __init__(
        self,
        type: str,
        allow_reset: bool = True,
        embedder_config: Any | None = None,
        crew: Any = None,
        qdrant_location: str | None = None,
        qdrant_api_key: str | None = None,
        sparse_embedder: Any | None = None,
        reranker: Any | None = None,
    ):
        self.type = type
        # 项目传入的是 config.embedder_config，一个把可调用的
        # EmbeddingFunction 放在 "provider" 键下的 dict（见 config.py）。
        assert embedder_config is not None
        self.embedder_config = embedder_config["provider"]
        self.sparse_embedder = sparse_embedder
        self.reranker = reranker
        self._qdrant_location = qdrant_location
        self._qdrant_api_key = qdrant_api_key
        self.app: QdrantClient = self._initialize_app()

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: dict | None = None,
    ) -> list[dict]:
        # 混合检索：两条并行预取路径（dense + sparse）经 RRF 融合，
        # 再对融合后的候选做交叉编码器重排。不设分数阈值——RRF 分数
        # 基于排名而非余弦相似度，为 dense 分数调的阈值没有意义。
        query = self._normalize_text(query)
        query_filter = self._to_qdrant_filter(filter)

        # 重排发生在 rerank_limit 个候选上，然后取前 limit 个。
        rerank_limit = max(limit * 4, 20) if self.reranker else limit

        if self.sparse_embedder is not None:
            dense = self.embedder_config([query])[0]
            sparse = self.sparse_embedder.embed([query])[0]
            response = self.app.query_points(
                self.type,
                prefetch=[
                    models.Prefetch(
                        query=dense,
                        using=DENSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=rerank_limit,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                        using=SPARSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=rerank_limit,
                    ),
                ],
                query=models.RrfQuery(
                    rrf=models.Rrf(k=RRF_K, weights=RRF_WEIGHTS)
                ),
                limit=rerank_limit,
            )
        else:
            embedding = self.embedder_config([query])[0]
            response = self.app.query_points(
                self.type,
                query=embedding,
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=rerank_limit,
            )
        results = [
            {
                "id": point.id,
                "metadata": (point.payload or {}).get("metadata"),
                "context": (point.payload or {}).get("value"),
                "score": point.score,
            }
            for point in response.points
        ]

        if self.reranker and len(results) > 1:
            scores = self.reranker.rerank(query, [r["context"] for r in results])
            for result, score in zip(results, scores, strict=True):
                result["rrf_score"] = result["score"]
                result["score"] = score
            results.sort(key=lambda r: r["score"], reverse=True)

        return results[:limit]

    def reset(self) -> None:
        self.app.delete_collection(self.type)

    def save(self, value: str, metadata: dict[str, Any]) -> None:
        """保存单条条目（实际逻辑见 save_batch）。"""
        self.save_batch([(value, metadata)])
    def save_batch(self, entries: list[tuple[str, dict[str, Any]]]) -> None:
        """
        一次性保存多条条目：所有文本做一次嵌入调用、一次 Qdrant upsert。
        比逐 chunk 往返快得多，尤其是对远程（Cloud）Qdrant。
        :param entries: (value, metadata) 元组列表
        """
        if not entries:
            return

        # 限制文档长度，避免超出模型的承受范围
        values = [self._normalize_text(value) for value, _ in entries]
        metadatas = [metadata for _, metadata in entries]

        # 一次性嵌入所有文本
        embeddings = self.embedder_config(values)
        vectors: list[dict[str, Any]] = [
            {DENSE_VECTOR_NAME: embedding} for embedding in embeddings
        ]
        if self.sparse_embedder is not None:
            sparse = self.sparse_embedder.embed(values)
            for vector, embedding in zip(vectors, sparse, strict=True):
                vector[SPARSE_VECTOR_NAME] = models.SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist(),
                )

        # 单个请求 upsert 所有点
        self.app.upsert(
            self.type,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload={"value": value, "metadata": metadata},
                )
                for vector, value, metadata in zip(vectors, values, metadatas, strict=True)
            ],
        )

    def delete(self, filter: dict | None = None) -> None:
        if filter is None:
            self.app.delete_collection(self.type)
            return
        self.app.delete(
            self.type,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value),
                        )
                        for key, value in filter.items()
                    ]
                ),
            ),
        )

    def count(self, filter: dict | None = None) -> int:
        return self.app.count(
            self.type,
            count_filter=self._to_qdrant_filter(filter),
        ).count

    def get_metadata_value(self, filter: dict, key: str) -> Any | None:
        """
        返回第一个匹配过滤器的点的指定 metadata 键的值，没有匹配点则
        返回 None。用于读取文件级的簿记字段（如 total_chunks），不必
        遍历整个集合。
        """
        points, _ = self.app.scroll(
            collection_name=self.type,
            scroll_filter=self._to_qdrant_filter(filter),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        metadata = (points[0].payload or {}).get("metadata") or {}
        return metadata.get(key)

    def list_src_paths(self) -> set[str]:
        """
        通过遍历所有点，返回集合中当前存储的不同 ``src_path`` 值的集合。
        """
        paths: set[str] = set()
        offset = None
        while True:
            points, offset = self.app.scroll(
                collection_name=self.type,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                metadata = (point.payload or {}).get("metadata") or {}
                src_path = metadata.get("src_path")
                if src_path:
                    paths.add(src_path)
            if offset is None:
                break
        return paths

    def _initialize_app(self) -> QdrantClient:
        # 初始化 Qdrant 客户端，集合不存在则创建
        client = QdrantClient(self._qdrant_location, api_key=self._qdrant_api_key)
        if not client.collection_exists(self.type):
            # 用一个哑值生成嵌入，以获得嵌入维度
            embedding = self.embedder_config([self.TEST_STRING])[0]

            # 即使只有 dense 也始终使用命名向量：save_batch 和 search
            # 都通过 DENSE_VECTOR_NAME 寻址 dense 向量。
            vectors_config: models.VectorsConfig = {
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=len(embedding),
                    distance=models.Distance.COSINE,
                )
            }
            sparse_vectors_config = None
            if self.sparse_embedder is not None:
                # IDF 修饰符给高频词降权，提升 BM25 在中文文本上的区分度。
                sparse_vectors_config = {
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(),
                        modifier=models.Modifier.IDF,
                    )
                }

            # 创建带 dense（以及可选 sparse）向量的 Qdrant 集合
            client.create_collection(
                collection_name=self.type,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )

            # 为过滤器用到的字段创建 payload 索引
            for field in (
                "metadata.src_path",
                "metadata.content_hash",
                "metadata.folder",
            ):
                client.create_payload_index(
                    collection_name=self.type,
                    field_name=field,
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD
                    ),
                )
        elif self.sparse_embedder is not None:
            # 防护混合检索之前创建的集合：向 dense-only 集合 upsert 混合
            # 点会在服务端报出令人困惑的错误，所以在这里快速失败并给出
            # 可操作的提示。
            existing = client.get_collection(self.type).config.params.sparse_vectors
            if SPARSE_VECTOR_NAME not in (existing or {}):
                raise RuntimeError(
                    f"Collection '{self.type}' predates hybrid search (no sparse "
                    f"'{SPARSE_VECTOR_NAME}' vector). Delete it and restart to "
                    "re-ingest the vault with dense+sparse vectors."
                )
        return client

    def _normalize_text(self, text: str) -> str:
        """
        把文本归一化到最大长度以内。
        :param text:
        :return:
        """
        encoded = text.encode("utf-8")
        if len(encoded) <= self.MAX_LENGTH_BYTES:
            return text

        # 按字节上限截断。errors="ignore" 解码会丢弃末尾不完整的多字节
        # 字符，而不是抛异常。
        truncated = encoded[: self.MAX_LENGTH_BYTES]
        return truncated.decode("utf-8", errors="ignore")

    def _to_qdrant_filter(self, filter: dict | None) -> models.Filter | None:
        """
        把 dict 过滤器转换成 Qdrant 过滤器。目前只支持精确匹配。
        :param filter:
        :return:
        """
        if filter is None:
            return None

        must = []
        for key, value in filter.items():
            must.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
            )
        return models.Filter(must=must)
