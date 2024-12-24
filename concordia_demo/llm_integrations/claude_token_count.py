import anthropic
from concordia_demo.environment import anthropic_api_key

_raw_client = anthropic.Anthropic(api_key=anthropic_api_key)


def claude_token_count(text: str, model: str = "claude-3-5-sonnet-latest") -> int:
    response = _raw_client.beta.messages.count_tokens(
        # betas=["token-counting-2024-11-01"],
        model=model,
        system="",
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens


if __name__ == "__main__":
    print(claude_token_count("hello my name is"))
