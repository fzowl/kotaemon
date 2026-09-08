"""Implements embeddings from [VoyageAI by MongoDB](https://voyageai.com)."""

import importlib

from kotaemon.base import Document, DocumentWithEmbedding, Param

from .base import BaseEmbeddings

vo = None

# Contextualized-chunk embedding models are served through the dedicated
# `contextualized_embed` API rather than the plain `embed` endpoint.
CONTEXTUALIZED_MODELS = ("voyage-context-4", "voyage-context-3")

# Upper bound (in tokens) for a single auto-produced chunk. Using the maximum
# supported context means each input string resolves to a single chunk (and thus
# a single embedding) unless it exceeds the model context window.
CONTEXTUALIZED_CHUNK_SIZE = 32000


def _import_voyageai():
    global vo
    if not vo:
        vo = importlib.import_module("voyageai")
    return vo


def _format_output(texts: list[str], embeddings: list[list]):
    """Formats the output of all `.embed` calls.
    Args:
        texts: List of original documents
        embeddings: Embeddings corresponding to each document
    """
    return [
        DocumentWithEmbedding(content=text, embedding=embedding)
        for text, embedding in zip(texts, embeddings)
    ]


class VoyageAIEmbeddings(BaseEmbeddings):
    """VoyageAI by MongoDB provides best-in-class embedding models and rerankers.

    Standard models (e.g. ``voyage-4-large``) go through the ``embed`` API.
    Contextualized-chunk models (``voyage-context-4``, ``voyage-context-3``) go
    through the ``contextualized_embed`` API: each input string is embedded as
    its own independent document. The batch is sent as a flat ``list[str]`` with
    ``enable_auto_chunking=True`` and ``chunk_size=32000`` so that every string
    resolves to exactly one chunk and one embedding. Cross-input
    contextualization is intentionally not used, because the generic embedding
    callers pass unrelated texts.
    """

    api_key: str = Param(None, help="Voyage API key", required=False)
    model: str = Param(
        "voyage-4-large",
        help=(
            "Model name to use. The VoyageAI by MongoDB "
            "[documentation](https://docs.voyageai.com/docs/embeddings) "
            "provides a list of all available embedding models. Current models "
            "include general-purpose `voyage-4-large`, `voyage-4`, "
            "`voyage-4-lite`; domain models `voyage-code-4`, `voyage-finance-2`, "
            "`voyage-law-2`; and contextualized-chunk models `voyage-context-4`, "
            "`voyage-context-3`."
        ),
        required=True,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.api_key:
            raise ValueError("API key must be provided for VoyageAIEmbeddings.")

        self._client = _import_voyageai().Client(api_key=self.api_key)
        self._aclient = _import_voyageai().AsyncClient(api_key=self.api_key)

    @property
    def _is_contextualized(self) -> bool:
        return self.model in CONTEXTUALIZED_MODELS

    def _contextualized_kwargs(self, texts: list[str], input_type: str | None) -> dict:
        """Build the kwargs for a `contextualized_embed` call.

        Auto-chunking is enabled for documents (the default). The API rejects
        auto-chunking for queries, so `input_type="query"` disables it and drops
        `chunk_size`.
        """
        enable_auto_chunking = input_type != "query"
        call_kwargs: dict = {
            "inputs": texts,
            "model": self.model,
            "enable_auto_chunking": enable_auto_chunking,
        }
        if input_type is not None:
            call_kwargs["input_type"] = input_type
        if enable_auto_chunking:
            call_kwargs["chunk_size"] = CONTEXTUALIZED_CHUNK_SIZE
        return call_kwargs

    def invoke(
        self,
        text: str | list[str] | Document | list[Document],
        *args,
        input_type: str | None = None,
        **kwargs,
    ) -> list[DocumentWithEmbedding]:
        texts = [t.content for t in self.prepare_input(text)]
        if self._is_contextualized:
            result = self._client.contextualized_embed(
                **self._contextualized_kwargs(texts, input_type)
            )
            embeddings = [r.embeddings[0] for r in result.results]
        else:
            embeddings = self._client.embed(texts, model=self.model).embeddings
        return _format_output(texts, embeddings)

    async def ainvoke(
        self,
        text: str | list[str] | Document | list[Document],
        *args,
        input_type: str | None = None,
        **kwargs,
    ) -> list[DocumentWithEmbedding]:
        texts = [t.content for t in self.prepare_input(text)]
        if self._is_contextualized:
            result = await self._aclient.contextualized_embed(
                **self._contextualized_kwargs(texts, input_type)
            )
            embeddings = [r.embeddings[0] for r in result.results]
        else:
            response = await self._aclient.embed(texts, model=self.model)
            embeddings = response.embeddings
        return _format_output(texts, embeddings)
