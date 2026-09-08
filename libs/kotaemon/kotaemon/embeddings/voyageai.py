"""Implements embeddings from VoyageAI by MongoDB (https://voyageai.com)."""

import importlib
from typing import Union

from kotaemon.base import Document, DocumentWithEmbedding, Param

from .base import BaseEmbeddings

vo = None


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

    This wrapper supports both the regular text embedding models (e.g.
    ``voyage-4-large``, ``voyage-4``, ``voyage-code-4``) and the
    contextualized-chunk embedding models (``voyage-context-4``). Contextualized
    models are routed through the ``contextualized_embed`` API instead of the
    plain ``embed`` API; the model name (prefix ``voyage-context``) decides which
    path is used.
    """

    api_key: str = Param(None, help="VoyageAI by MongoDB API key", required=False)
    model: str = Param(
        "voyage-4-large",
        help=(
            "Model name to use. The VoyageAI by MongoDB "
            "[documentation](https://docs.voyageai.com/docs/embeddings) lists all "
            "available embedding models. Current general-purpose models are "
            "`voyage-4-large`, `voyage-4` and `voyage-4-lite`; domain-specific "
            "models include `voyage-code-4`, `voyage-finance-2` and "
            "`voyage-law-2`. The contextualized-chunk models (e.g. "
            "`voyage-context-4`) are documented "
            "[here](https://docs.voyageai.com/docs/contextualized-chunk-embeddings)."
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
        """Whether ``self.model`` is a contextualized-chunk embedding model."""
        return self.model.startswith("voyage-context")

    @staticmethod
    def _collect_contextualized(response) -> list[list]:
        """Return one embedding per input, in input order.

        The ``contextualized_embed`` response holds one result per top-level
        input, each with a list of per-chunk ``embeddings`` and an ``index``
        pointing back to the input position. Inputs are built as single-chunk
        documents (see :meth:`_contextualized_embed`), so the first embedding of
        each result is the vector for that input.
        """
        results = sorted(response.results, key=lambda r: r.index)
        return [result.embeddings[0] for result in results]

    def _contextualized_embed(self, texts: list[str]) -> list[list]:
        # Per the official spec the ``inputs`` argument is
        # ``Union[List[List[str]], List[str]]``: a flat ``list[str]`` (each
        # string a standalone document) or a nested ``list[list[str]]`` (each
        # inner list the pre-split chunks of one document). Both formats are
        # accepted by the API; we use the nested single-chunk form so every
        # input text maps to exactly one output embedding.
        # https://docs.voyageai.com/docs/contextualized-chunk-embeddings
        inputs: Union[list[list[str]], list[str]] = [[text] for text in texts]
        response = self._client.contextualized_embed(inputs=inputs, model=self.model)
        return self._collect_contextualized(response)

    async def _acontextualized_embed(self, texts: list[str]) -> list[list]:
        inputs: Union[list[list[str]], list[str]] = [[text] for text in texts]
        response = await self._aclient.contextualized_embed(
            inputs=inputs, model=self.model
        )
        return self._collect_contextualized(response)

    def invoke(
        self, text: str | list[str] | Document | list[Document], *args, **kwargs
    ) -> list[DocumentWithEmbedding]:
        texts = [t.content for t in self.prepare_input(text)]
        if not texts:
            return []

        if self._is_contextualized:
            embeddings = self._contextualized_embed(texts)
        else:
            embeddings = self._client.embed(texts, model=self.model).embeddings
        return _format_output(texts, embeddings)

    async def ainvoke(
        self, text: str | list[str] | Document | list[Document], *args, **kwargs
    ) -> list[DocumentWithEmbedding]:
        texts = [t.content for t in self.prepare_input(text)]
        if not texts:
            return []

        if self._is_contextualized:
            embeddings = await self._acontextualized_embed(texts)
        else:
            response = await self._aclient.embed(texts, model=self.model)
            embeddings = response.embeddings
        return _format_output(texts, embeddings)
