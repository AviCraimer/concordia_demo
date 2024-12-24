from concordia_demo.llm_integrations.claude_client import claude_client


def similarity_prompt(text1: str, text2: str):
    return f"""On a scale of 0.0 to 1.0, how semantically similar are these two texts?

    <text-1>
    {text1}
    </text-1>

    <text-2>
    {text2}
    </text-2>

    Respond with just the number.
"""


def semantic_similarity(text1: str, text2: str) -> float:

    response = claude_client.get_response([], similarity_prompt(text1, text2))

    score = float(response.strip())
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Score {score} must be between 0.0 and 1.0")
    return score
