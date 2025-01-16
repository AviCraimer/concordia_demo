# config.py or environment.py
import os
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()


def get_required_env_var(env_var_name: str) -> str:
    value = os.getenv(env_var_name)
    if value is None:
        raise ValueError(f"Required environment variable {env_var_name} is not set")
    return value


# openai_api_key = get_required_env_var("OPENAI_API_KEY")
anthropic_api_key = get_required_env_var("ANTHROPIC_API_KEY")
