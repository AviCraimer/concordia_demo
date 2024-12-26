"""Claude language model implementation for Concordia."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any
import random
from concordia.language_model.language_model import (
    LanguageModel,
    InvalidResponseError,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TERMINATORS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
)

from concordia_demo.llm_integrations.claude_client import claude_client


class ClaudeModel(LanguageModel):
    """Claude implementation of the Concordia language model interface."""

    def __init__(self, client=claude_client):
        self.client = client

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = DEFAULT_TERMINATORS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        """Samples text from Claude.

        Returns:
            The sampled response (does not include the prompt).

        Raises:
            TimeoutError: If the operation times out.
        """

        try:
            response = self.client.get_response(
                messages=[],
                system_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Handle terminators if any are specified
            if terminators:
                for terminator in terminators:
                    if terminator in response:
                        response = response[: response.index(terminator)]

            return response.strip()

        except Exception as e:
            raise TimeoutError(f"Claude API request failed: {str(e)}")

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        """Simple random choice implementation.
        We aren't using this yet, it is just a placeholder."""
        if seed is not None:
            random.seed(seed)

        idx = random.randrange(len(responses))
        return idx, responses[idx], {}
