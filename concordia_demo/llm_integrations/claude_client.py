import anthropic
from anthropic.types import Message, MessageParam, TextBlock, ContentBlock

from typing import Tuple
from concordia_demo.environment import anthropic_api_key


type ChatMessage = dict[str, str]  # {"role": str, "content": str}


# Core Claude interaction
class ClaudeClient:

    def __init__(
        self, api_key: str = anthropic_api_key, model: str = "claude-3-5-sonnet-latest"
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    @classmethod
    def clean_messages_(cls, history: list[dict[str, str]]) -> list[MessageParam]:
        """
        Ensures only user and assistant messages are in message list.
        """
        cleaned_messages = []

        claude_messages: list[MessageParam] = [
            {
                "content": msg["content"],
                "role": "assistant" if msg.get("role") == "assistant" else "user",
            }
            for msg in history
            if msg["role"] in ["user", "assistant"]
        ]

        return claude_messages

    def handle_empty_msgs(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        # Claude requires a non-empty array of messages
        default_messages = [
            {
                "role": "user",
                "content": "Please follow the instructions in you system prompt now.",
            }
        ]

        if not messages:
            return default_messages
        else:
            return messages

    def get_response(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:

        try:

            messages = self.handle_empty_msgs(messages)

            cleaned_messages = self.__class__.clean_messages_(messages)

            default_model = self.model

            content: ContentBlock = self.client.messages.create(
                model=default_model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system=system_prompt,
                messages=cleaned_messages,
            ).content[0]

            assert isinstance(content, TextBlock)
            return content.text
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")


# A ready made instance of ClaudeClient, for any text completion task that doesn't need special options this can be used.
claude_client = ClaudeClient()
