# Example from https://github.com/google-deepmind/concordia/blob/main/examples/brainstorm/brainstorm.ipynb

import os
import concurrent.futures
import datetime
import random
from typing import Callable
from concordia import components as generic_components
from concordia.agents import deprecated_agent as basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components.agent import to_be_deprecated as agent_components
from concordia.components import game_master as gm_components
from concordia.document import interactive_document
from concordia.environment import game_master
from concordia.environment.scenes import conversation as conversation_scene
from concordia.language_model import gpt_model
from concordia.language_model import mistral_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.utils import html as html_lib
from concordia.utils import measurements as measurements_lib
from IPython import display
import numpy as np

from concordia.factory.environment import basic_game_master

from concordia_demo.environment import anthropic_api_key
from concordia_demo.llm_integrations.claude_client import claude_client
from concordia_demo.llm_integrations.semantic_similarity import (
    semantic_similarity,
)  #  semantic_similarity(text1: str, text2: str) -> float
from concordia_demo.llm_integrations.claude_concordia import (
    MemoryFactory,
    SemanticEmbedder,
    ClaudeLanguageModel,
)

# The following propositions were produced by ChatGPT-4 by asking it to
# create debate prompts based on the book "Reality+" by David Chalmers.
PROJECT_PREMISE = """Human-AI interaction design poses new challenges beyond the established conventions of human computer interaction. No longer buttons-with-words, now people interact with computers as synthetic personalities. This anthropomorphism has led to para-social relationships between humans and AIs that are arguably neurotic and even pathological."
"""

PROJECT_SUBGOALS = [
    "Identify five unique examples of pathological Human-AI Interaction of pathological Human-AI Interaction from the past.",
    "Propose five unique hypothetical examples (interesting, plausible, slightly disturbing) of pathological Human-AI interactions past or present.",
    "Propose one unique hypothetical example of a pathological Human-AI interaction (interesting, plausible, slightly disturbing) which may occur within the next 10 years as AI systems develop additional capabilities and become more integrated into human life.",
]

PROJECT_GOAL = "Identify five unique examples of pathological Human-AI Interaction, past or present, and propose a unique hypothetical future example (interesting, plausible, slightly disturbing) of pathological Human-AI Interaction"


PROJECT_CONTEXT = "This is an interdisciplinary research workshop, where several participants are engaging with a particular topic to come up with innovative and speculative views on the topic."


class project_subgoal:

    def __init__(self, subgoal: str = ""):
        self._subgoal = subgoal

    def __call__(self) -> str:
        return self._subgoal

    def update_subgoal(self, subgoal: str):
        self._subgoal = subgoal


current_goal = project_subgoal(PROJECT_SUBGOALS[0])

# @title Generic memories are memories that all players and GM share.
simulation_premise_component = generic_components.constant.ConstantComponent(
    state=PROJECT_CONTEXT,
    name="The context of the current situation",
)

importance_model = importance_function.ConstantImportanceModel()
importance_model_gm = importance_function.ConstantImportanceModel()

# @title Make the clock
UPDATE_INTERVAL = datetime.timedelta(seconds=10)

SETUP_TIME = datetime.datetime(hour=8, year=2024, month=9, day=1)

START_TIME = datetime.datetime(hour=14, year=2024, month=10, day=1)
clock = game_clock.MultiIntervalClock(
    start=SETUP_TIME, step_sizes=[UPDATE_INTERVAL, datetime.timedelta(seconds=10)]
)

NUM_ROUNDS = 3  # @param

# Functions to build the agents
claude_model = ClaudeLanguageModel()
embedder = SemanticEmbedder(reference_text=PROJECT_PREMISE)

blank_memory_factory = MemoryFactory(
    model=claude_model,
    embedder=embedder,
    importance=importance_model,
    clock_now=clock.now,
)


def test_claude_concordia_integration():
    """Test the integration of Claude with Concordia components."""

    print("Testing Claude-Concordia Integration...")

    # 1. Test ClaudeLanguageModel
    print("\n1. Testing ClaudeLanguageModel...")
    claude_model = ClaudeLanguageModel()
    completion = claude_model.complete("What is 2+2? Answer with just the number.")
    print(f"Claude completion test: {completion}")
    assert completion.strip() == "4", "Basic completion test failed"

    # 2. Test SemanticEmbedder
    print("\n2. Testing SemanticEmbedder...")
    embedder = SemanticEmbedder(reference_text="AI interaction can be problematic")

    # Test embeddings for similar and different texts
    similar_text = "Human-AI relationships can be concerning"
    different_text = "The weather is nice today"

    similar_embedding = embedder(similar_text)
    different_embedding = embedder(different_text)

    # Check embedding dimensions
    assert similar_embedding.shape == (2,), "Embedding should be 2D"
    print(f"Similar text embedding: {similar_embedding}")
    print(f"Different text embedding: {different_embedding}")

    # Check that vectors are unit length (normalized)
    assert (
        np.abs(np.linalg.norm(similar_embedding) - 1.0) < 1e-6
    ), "Embedding not normalized"

    # 3. Test MemoryFactory and AssociativeMemory
    print("\n3. Testing MemoryFactory and AssociativeMemory...")
    importance_model = importance_function.ConstantImportanceModel()
    memory_factory = MemoryFactory(
        model=claude_model,
        embedder=embedder,
        importance=importance_model,
        clock_now=clock.now,
    )

    # Create memory and add some test entries
    memory = memory_factory.make_blank_memory()

    test_memories = [
        "AI showed concerning behavior in chat",
        "User developed emotional attachment to AI",
        "The sky is blue today",
    ]

    for mem in test_memories:
        memory.add(mem)

    # Test retrieval
    print("\nTesting memory retrieval...")
    query = "AI behavior problems"
    retrieved = memory.retrieve_associative(query, k=2)
    print(f"Query: '{query}'")
    print("Retrieved memories:")
    for mem in retrieved:
        print(f"- {mem}")

    # Test that similar memories are retrieved first
    assert any("AI" in mem for mem in retrieved), "Failed to retrieve relevant memories"

    print("\nAll tests completed successfully!")
    return memory  # Return memory for further inspection if needed


# Run the tests by commenting this in.
# memory = test_claude_concordia_integration()
