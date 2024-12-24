# claude_concordia
# Integration of claude LLM with concordia library

from datetime import datetime
from typing import Callable
import numpy as np
from concordia.associative_memory import (
    importance_function,
    associative_memory,
)
from concordia_demo.llm_integrations.claude_client import claude_client
from concordia_demo.llm_integrations.semantic_similarity import semantic_similarity


class ClaudeLanguageModel:
    """
    Adapter class to make Claude client compatible with Concordia's language model interface.

    Concordia expects language models to have a complete() method that takes a prompt and returns text.
    This wrapper provides that interface for our Claude client.

    Similar to concordia.language_model.LanguageModel base class but simplified for our needs.
    """

    def __init__(self, client=claude_client):
        self.client = client

    def complete(self, prompt: str) -> str:
        """
        Complete a prompt using Claude.

        Args:
            prompt: The text prompt to complete

        Returns:
            The completion text from Claude
        """

        return self.client.get_response([], prompt)


class SemanticEmbedder:
    """
    Text embedder that creates embeddings based on semantic similarity. Concordia uses embeddings to find related memories in AssociativeMemory.

    The embedding approach:
    1. Calculates semantic similarity between input text and reference text (0.0 to 1.0)
    2. Maps similarity to angle between 0 and π/2 radians
    3. Creates 2D unit vector using this angle

    This ensures:
    - Cosine similarity between vectors will reflect semantic similarity
    - Vectors are normalized (length 1) for proper similarity calculations
    - Compatible with AssociativeMemory's vector operations

    Example:
        similarity 1.0 -> angle π/2 -> vector [0, 1]
        similarity 0.0 -> angle 0   -> vector [1, 0]
    """

    def __init__(self, reference_text: str = ""):
        self.reference_text = reference_text

    def __call__(self, text: str) -> np.ndarray:
        """
        Convert text to a normalized 2D embedding vector.

        Args:
            text: The text to embed

        Returns:
            2D unit vector where angle represents similarity to reference text
        """
        similarity = semantic_similarity(text, self.reference_text)
        # Convert similarity [0,1] to angle [0,π/2]
        angle = similarity * np.pi / 2
        # Return normalized 2D vector
        return np.array([np.cos(angle), np.sin(angle)])


class MemoryFactory:
    """
    Factory class for creating blank associative memories.

    This is a simplified version of concordia.associative_memory.blank_memories.MemoryFactory
    that works with our Claude-based components instead of the original GPT/sentence-transformer setup.

    The factory provides consistent configuration for creating AssociativeMemory instances
    with the same language model, embedder, importance function, and clock.
    """

    def __init__(
        self,
        model: ClaudeLanguageModel,
        embedder: Callable[[str], np.ndarray],
        importance: importance_function.ImportanceModel,
        clock_now: Callable[[], datetime] | None = None,
    ):
        """
        Args:
            model: Language model for text generation
            embedder: Function to convert text to vector embeddings
            importance: Function to score memory importance (0 to 1)
            clock_now: Function to get current time for memory timestamps
        """
        self._model = model
        self._embedder = embedder
        self._importance = importance.importance  # Get the bound method
        self._clock_now = clock_now or datetime.now

    def make_blank_memory(self) -> associative_memory.AssociativeMemory:
        """
        Creates a new empty AssociativeMemory instance.

        Returns:
            A fresh AssociativeMemory configured with this factory's components
        """
        return associative_memory.AssociativeMemory(
            self._embedder,
            self._importance,
            clock=self._clock_now,
        )


# # Example usage:
# claude_model = ClaudeLanguageModel()
# embedder = SemanticEmbedder(reference_text=PROJECT_PREMISE)

# blank_memory_factory = MemoryFactory(
#     model=claude_model,
#     embedder=embedder,
#     importance=importance_model.importance,
#     clock_now=clock.now,
# )
