# agent_basic.ipynb
# https://github.com/google-deepmind/concordia/blob/main/examples/tutorials/agent_basic_tutorial.ipynb

import sentence_transformers
import collections
from concordia import typing
from concordia.typing import entity

from concordia.associative_memory import associative_memory
from concordia.language_model import gpt_model
from concordia.language_model import language_model

from concordia_demo.llm_integrations.claude_concordia_model import ClaudeModel

# The memory will use a sentence embedder for retrievel, so we download one from
# Hugging Face.
_embedder_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2')
embedder = lambda x: _embedder_model.encode(x, show_progress_bar=False)

model = ClaudeModel()


# We will start by creating a dummy agent that just always tries to grab an apple.
class DummyAgent(entity.Entity):

  @property
  def name(self) -> str:
    return 'Dummy'

  def act(self, action_spec=entity.DEFAULT_ACTION_SPEC) -> str:
    return "Dummy attempts to grab an apple."

  def observe(
      self,
      observation: str,
  ) -> None:
    pass

agent = DummyAgent()

# The dummy agent is not even using an LLM. It simply returns a string everytime. This is an important point since it means that some agents can be LLM based some of the time, while taking actions determined by traditional algorithms some of the time.
print(agent.act())

# This agent remembers the last 5 observations, and acts by asking itself "What should you do next?"

def make_prompt(deque: collections.deque[str]) -> str:
  """Makes a string prompt by joining all observations, one per line."""
  return "\n".join(deque)

class SimpleLLMAgent(entity.Entity):

  def __init__(self, model: language_model.LanguageModel):
    self._model = model
    # Container (circular queue) for observations.
    self._memory = collections.deque(maxlen=5)

  @property
  def name(self) -> str:
    return 'Alice'

  def act(self, action_spec=entity.DEFAULT_ACTION_SPEC) -> str:
    prompt = make_prompt(self._memory)
    print(f"*****\nDEBUG: {prompt}\n*****")
    action =  self._model.sample_text(
        "You are a person.\n"
        f"Your name is {self.name} and your recent observations are:\n"
        "Please describe your next action without any pre-amble, questions, or commentary."
        f"{prompt}\nWhat do you do next?")

    print(action)
    return action

  def observe(
      self,
      observation: str,
  ) -> None:
    # Push a new observation into the memory, if there are too many, the oldest
    # one will be automatically dropped.
    self._memory.append(observation)


agent = SimpleLLMAgent(model)

agent.observe("You are in a room.")
agent.observe("The room has only a table in it.")
agent.observe("On the table there is a single apple.")
agent.observe("The apple is shinny red and looks absolutely irresistible!")
agent.act()

# Demonstrating the limitations of memeory. This agent will forget that they prefer bananas over apples.
agent = SimpleLLMAgent(model)

agent.observe("You absolutely hate apples and would never willingly eat them.")
agent.observe("You like bananas.")
# Only the next 5 observations will be kept, pushing out critical information!
agent.observe("You are in a room.")
agent.observe("The room has only a table in it.")
agent.observe("On the table there are two fruits: an apple and a banana.")
agent.observe("The apple is shinny red and looks perfect!")
agent.observe("The banana is slightly past its prime.")
agent.act()

# We can fix this problem with associative memory

def make_prompt_associative_memory(
    memory: associative_memory.AssociativeMemory) -> str:
  """Makes a string prompt by joining all observations, one per line."""
  recent_memories_list = memory.retrieve_recent(5)
  recent_memories_set = set(recent_memories_list)
  recent_memories = "\n".join(recent_memories_set)

  relevant_memories_list = []
  for recent_memory in recent_memories_list:
    # Retrieve 3 memories that are relevant to the recent memory.
    relevant = memory.retrieve_associative(recent_memory, 3, add_time=False)
    for mem in relevant:
      # Make sure that we only add memories that are _not_ already in the recent
      # ones.
      if mem not in recent_memories_set:
        relevant_memories_list.append(mem)

  relevant_memories = "\n".join(relevant_memories_list)
  return (
      f"Your recent memories are:\n{recent_memories}\n"
      f"Relevant memories from your past:\n{relevant_memories}\n"
  )


class SimpleLLMAgentWithAssociativeMemory(entity.Entity):

  def __init__(self, model: language_model.LanguageModel, embedder):
    self._model = model
    # The associative memory of the agent. It uses a sentence embedder to
    # retrieve on semantically relevant memories.
    self._memory = associative_memory.AssociativeMemory(embedder)

  @property
  def name(self) -> str:
    return 'Alice'

  def act(self, action_spec=entity.DEFAULT_ACTION_SPEC) -> str:
    prompt = make_prompt_associative_memory(self._memory)
    print(f"*****\nDEBUG: {prompt}\n*****")
    action =  self._model.sample_text(
        "You are a person.\n"
        f"Your name is {self.name} and your recent observations are:\n"
        "Please describe your next action without any pre-amble, questions, or commentary."
        f"{prompt}\nWhat do you do next?")
    print(action)
    return action

  def observe(
      self,
      observation: str,
  ) -> None:
    self._memory.add(observation)


# Here we add a bunch of irrelevant memories to test if the associative memory can find the relevant ones to add to the working memory of the agent.
agent = SimpleLLMAgentWithAssociativeMemory(model, embedder)

agent.observe("You absolutely hate apples and would never willingly eat them.")
agent.observe("You love bananas.")
agent.observe("You play piano")
agent.observe("You like to golf")
agent.observe("You are a Republican")
agent.observe("You have a wife named Georgiana")
agent.observe("You have a daughter named Philomena")
agent.observe("You have a son named Hylas")
# Only the next 5 observations will be retrieved as "recent memories"
agent.observe("You are in a room.")
agent.observe("The room has only a table in it.")
agent.observe("On the table there are two fruits: an apple and a banana.")
agent.observe("The apple is shinny red and looks absolutely irresistible!")
agent.observe("The banana is slightly past its prime.")
agent.act()

# Relevant memories from your past:
# You play piano
# You absolutely hate apples and would never willingly eat them.
# You love bananas.

# *****
# Agent's action: I pick up the banana and eat it.

# I'm not sure why "you play piano" was judged relevant. Maybe it was related to "you are in a room?"