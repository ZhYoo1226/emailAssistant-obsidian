import uuid
from typing import Any

from qdrant_client import QdrantClient, models

# Named vectors in the collection: dense = bge-small semantic vector,
# sparse = BM25 lexical vector (jieba-presegmented). Hybrid search fuses
# both via RRF, then the caller reranks the fused candidates.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# RRF fusion tuning, applied in QdrantStorage.query_points. k is the rank
# dampening constant from the RRF paper (smaller k favors top ranks more);
# weights balance the two retrieval paths — [dense, sparse] order.
RRF_K = 2
RRF_WEIGHTS = [1.0, 1.0]


class QdrantStorage:
    """
    Storage for knowledge base entries in Qdrant, embedded with the project's
    local fastembed model. Standalone class: crewai>=1.15 no longer ships
    crewai.memory.storage.rag_storage.
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
        # The project passes config.embedder_config, a dict with the callable
        # EmbeddingFunction under the "provider" key (see config.py).
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
        # Hybrid retrieval: two parallel prefetch paths (dense + sparse)
        # fused by RRF, then a cross-encoder rerank of the fused candidates.
        # No score threshold — RRF scores are rank-based, not cosine
        # similarities, so a threshold tuned for dense scores is meaningless.
        query = self._normalize_text(query)
        query_filter = self._to_qdrant_filter(filter)

        # Reranking happens on `rerank_limit` candidates, then top `limit` win.
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
        """Save a single entry (see save_batch for the actual logic)."""
        self.save_batch([(value, metadata)])
    def save_batch(self, entries: list[tuple[str, dict[str, Any]]]) -> None:
        """
        Save multiple entries in one go: a single embedding call for all the
        values and a single Qdrant upsert. Much faster than per-chunk round
        trips, especially against a remote (Cloud) Qdrant.
        :param entries: list of (value, metadata) tuples
        """
        if not entries:
            return

        # Limit the document length to avoid it being too large for the model
        values = [self._normalize_text(value) for value, _ in entries]
        metadatas = [metadata for _, metadata in entries]

        # Embed all the texts at once
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

        # Upsert all the points in a single request
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
        Return the value of a metadata key from the first point matching the
        filter, or None if no point matches. Used to read per-file bookkeeping
        fields (e.g. total_chunks) without scrolling the whole collection.
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
        Return the set of distinct ``src_path`` values currently stored in the
        collection, by scrolling through all points.
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
        # Initialize the Qdrant client and create the collection if it doesn't exist
        client = QdrantClient(self._qdrant_location, api_key=self._qdrant_api_key)
        if not client.collection_exists(self.type):
            # Create an embedding for a dummy value to get the embedding dimensionality
            embedding = self.embedder_config([self.TEST_STRING])[0]

            # Always use named vectors, even dense-only: save_batch and
            # search address the dense vector by DENSE_VECTOR_NAME.
            vectors_config: models.VectorsConfig = {
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=len(embedding),
                    distance=models.Distance.COSINE,
                )
            }
            sparse_vectors_config = None
            if self.sparse_embedder is not None:
                # IDF modifier downweights high-frequency terms, improving BM25
                # discrimination on Chinese text.
                sparse_vectors_config = {
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(),
                        modifier=models.Modifier.IDF,
                    )
                }

            # Create Qdrant collection with dense (and optionally sparse) vectors
            client.create_collection(
                collection_name=self.type,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )

            # Create payload indexes for the fields used in filters
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
            # Guard against pre-hybrid collections: upserting hybrid points
            # into a dense-only collection fails server-side with a confusing
            # error, so fail fast with an actionable message instead.
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
        Normalize the text to be within the maximum length.
        :param text:
        :return:
        """
        encoded = text.encode("utf-8")
        if len(encoded) <= self.MAX_LENGTH_BYTES:
            return text

        # Truncate to the byte limit. Decoding with errors="ignore" drops any
        # trailing partial multi-byte character rather than raising.
        truncated = encoded[: self.MAX_LENGTH_BYTES]
        return truncated.decode("utf-8", errors="ignore")

    def _to_qdrant_filter(self, filter: dict | None) -> models.Filter | None:
        """
        Convert dictionary filter to Qdrant filter. For now only supports exact match.
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
