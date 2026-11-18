"""Optional cross-encoder reranking for a bounded candidate set."""
from functools import cached_property


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, max_length=512)

    def rank(self, question, matches):
        if not matches:
            return matches
        scores = self.model.predict(
            [(question, match.document.page_content) for match in matches],
            batch_size=8, show_progress_bar=False,
        )
        return [match for _, match in sorted(zip(scores, matches), key=lambda item: item[0], reverse=True)]
