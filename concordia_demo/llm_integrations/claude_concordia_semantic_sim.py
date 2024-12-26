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
from concordia_demo.llm_integrations.claude_concordia_model import ClaudeModel


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


# # Example usage:
# claude_model = ClaudeLanguageModel()
# embedder = SemanticEmbedder(reference_text=PROJECT_PREMISE)
