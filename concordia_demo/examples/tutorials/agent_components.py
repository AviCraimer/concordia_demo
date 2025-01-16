# https://github.com/google-deepmind/concordia/blob/main/examples/tutorials/agent_components_tutorial.ipynb

import collections
from typing import TypedDict
import sentence_transformers

from concordia import typing
from concordia.typing import entity

from concordia.agents import entity_agent
from concordia.associative_memory import associative_memory

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component

from concordia.language_model import gpt_model
from concordia.language_model import language_model

from concordia_demo.llm_integrations.claude_concordia_model import ClaudeModel

# The memory will use a sentence embedder for retrievel, so we download one from
# Hugging Face.
_embedder_model = sentence_transformers.SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)
embedder = lambda x: _embedder_model.encode(x, show_progress_bar=False)

model = ClaudeModel()

# What is a Component?
# Recall that Entities have an act and an observe function they need to implement.

# A Component is just a modular piece of functionality that helps the agent make decisions to process the observations it receives, and to create action attempts. The EntityAgent is in charge of dispatching the action requests to its components, and to inform them once an action attempt has been decided. Likewise, the EntityAgent will inform components of observations received, and process any observation processing context from components.

# The minimal agent
# At the very least, an EntityAgent needs a special component called an ActingComponent which decides the action attempts. Let's create an ActingComponent that always tries eating an apple. We will then added to a dummy EntityAgent.


class AppleEating(entity_component.ActingComponent):

    def get_action_attempt(
        self,
        context,
        action_spec,
    ) -> str:
        return "Eat the apple."


# At a minimum we must provide the `act-component` to the EntityAgent
agent = entity_agent.EntityAgent("Alice", act_component=AppleEating())

print(agent.act())


# This is a very simple agent... it just always tries to eat the apple. So, let's make that a bit more interesting.

# Like we did in the Basic Tutorial, let's give the agent a memory, and make it decide what to do based on relevant memories to observations. Unlike the previous tutorial where we used an AssociativeMemory directly, we will use a memory component instead. This highlights the modularity of the component system.

# We will create a Component that received observations and pushes them into a memory Component. Then, we will create a Component that extracts recent memories. Finally, we will define an ActingComponent that takes context from all components, and produces an action attempt that is relevant to the situation.


class Observe(entity_component.ContextComponent):

    def pre_observe(self, observation: str) -> str:
        self.get_entity().get_component(
            "memory", type_=memory_component.MemoryComponent
        ).add(observation, {})
        return ""


class RecentMemories(entity_component.ContextComponent):

    def pre_act(self, action_spec) -> str:
        recent_memories_list = (
            self.get_entity()
            .get_component("memory", type_=memory_component.MemoryComponent)
            .retrieve(
                query="",  # Don't need a query to retrieve recent memories.
                limit=5,
                scoring_fn=legacy_associative_memory.RetrieveRecent(),
            )
        )
        recent_memories = " ".join(memory.text for memory in recent_memories_list)
        print(f"*****\nDEBUG: Recent memories:\n  {recent_memories}\n*****")
        return recent_memories


class SimpleActing(entity_component.ActingComponent):

    def __init__(self, model: language_model.LanguageModel):
        self._model = model

    def get_action_attempt(
        self,
        context: entity_component.ComponentContextMapping,
        action_spec,
    ) -> str:
        # Put context from all components into a string, one component per line.
        context_for_action = "\n".join(
            f"{name}: {context}" for name, context in context.items()
        )
        print(f"*****\nDEBUG:\n  context_for_action:\n{context_for_action}\n*****")
        # Ask the LLM to suggest an action attempt.
        call_to_action = action_spec.call_to_action.format(
            name=self.get_entity().name, timedelta="2 minutes"
        )
        sampled_text = self._model.sample_text(
            f"{context_for_action}\n\n{call_to_action}\n",
        )
        return sampled_text


raw_memory = legacy_associative_memory.AssociativeMemoryBank(
    associative_memory.AssociativeMemory(embedder)
)

# Let's create an agent with the above components.
agent = entity_agent.EntityAgent(
    "Alice",
    act_component=SimpleActing(model),
    context_components={
        "observation": Observe(),
        "recent_memories": RecentMemories(),
        "memory": memory_component.MemoryComponent(raw_memory),
    },
)

agent.observe("You absolutely hate apples and would never willingly eat them.")
agent.observe("You don't particularly like bananas.")
# Only the next 5 observations will be kept, pushing out critical information!
agent.observe("You are in a room.")
agent.observe("The room has only a table in it.")
agent.observe("On the table there are two fruits: an apple and a banana.")
agent.observe("The apple is shinny red and looks absolutely irresistible!")
agent.observe("The banana is slightly past its prime.")

print(agent.act())


# Alright! We have now have an agent that can use a very limited memory to choose actions :)

# A few things of notice in the definitions above.
# Some components are defining pre_act while others are defining pre_observe
# Acting components receive a contexts parameter
# Some components are finding other components within the agent via self.get_entity().get_component(component_name)

# Avi: So, context components are related to interfacing between some kind of state and the agent's actions and observations. In this case we use it to interface between a long term memory and the working memory that is put into the agent's context. However, it could also be used to interface with simulations of the world state, or whatever else.

# In summary, context components are more related to the game master role and action components are more related to the player agents.

# **The EntityComponent API**
# EntityComponents have the following functions, which you can override in your component implementation:

# pre_act(action_spec): Returns the information that the component wants to be part of the acting decision
# post_act(action_attempt): Informs component of the action decided by the ActingComponent. Returns any information that might be useful for the agent (usually empty)
# pre_obeserve(observation): Informs component of the observation received. Returns any information that might be useful for the agent (usually empty)
# post_observe(): Returns any information that might be useful for the agent (usually empty)
# update(): Inform the component that an act or observe are being finalized. Called after post_act or post_observe to give the component a chance to update its internal state
# These functions correspond to the Phases that an EntityAgent can be in. We will talk about Phases below.

# For more detailed information, see the definition of the EntityComponent and the EntityAgent

# **The ActingComponent API**
# ActingComponents have only one required function:

# get_action_attempt(context, action_spec): The contexts are a dictionary of component name to the returned values from all (entity/context) components' pre_act.
# So we can reference different context inputs in the acting component via the names of the names of the context components.
# Presumably we could also narrow the type of `context` param to ensure that the action component recieves the necessary info in its argument. However, when making a generic acting component class like this it might be based to handle all action contexts uniformly and leave the specific logic to individual context components.
# The ActingComponent then uses the contexts from the components and the action spec to decide on the action attempt. This action attempt will then be forwarded by the EntityAgent to all components via post_act.

# I'm thinking about the issue of how to ensure that context components have their dependencies met? In this example, Observe and RecentMemories both require that the entity they are attached to has a MemoryComponent named 'memory'. We could do a runtime check to ensure that this component is present, but it would be nice to find a way to enforce this with static types.


class RecentMemoriesImproved(action_spec_ignored.ActionSpecIgnored):

    def __init__(self):
        super().__init__("Recent memories")

    def _make_pre_act_value(self) -> str:
        recent_memories_list = (
            self.get_entity()
            .get_component("memory", type_=memory_component.MemoryComponent)
            .retrieve(
                query="",  # Don't need a query to retrieve recent memories.
                limit=5,
                scoring_fn=legacy_associative_memory.RetrieveRecent(),
            )
        )
        recent_memories = " ".join(memory.text for memory in recent_memories_list)
        print(f"*****\nDEBUG: Recent memories:\n  {recent_memories}\n*****")
        return recent_memories


def _recent_memories_str_to_list(recent_memories: str) -> list[str]:
    # Split sentences, strip whitespace and add final period
    return [memory.strip() + "." for memory in recent_memories.split(".")]


class RelevantMemories(action_spec_ignored.ActionSpecIgnored):

    def __init__(self):
        super().__init__("Relevant memories")

    def _make_pre_act_value(self) -> str:
        recent_memories = (
            self.get_entity().get_component("recent_memories").get_pre_act_value()
        )
        # Each sentence will be used for retrieving new relevant memories.
        recent_memories_list = _recent_memories_str_to_list(recent_memories)
        recent_memories_set = set(recent_memories_list)
        memory = self.get_entity().get_component("memory")
        relevant_memories_list = []
        for recent_memory in recent_memories_list:
            # Retrieve 3 memories that are relevant to the recent memory.
            relevant = memory.retrieve(
                query=recent_memory,
                limit=3,
                scoring_fn=legacy_associative_memory.RetrieveAssociative(
                    add_time=False
                ),
            )
            for mem in relevant:
                # Make sure that we only add memories that are _not_ already in the recent
                # ones.
                if mem.text not in recent_memories_set:
                    relevant_memories_list.append(mem.text)
                    recent_memories_set.add(mem.text)

        relevant_memories = "\n".join(relevant_memories_list)
        print(f"*****\nDEBUG: Relevant memories:\n{relevant_memories}\n*****")
        return relevant_memories


raw_memory = legacy_associative_memory.AssociativeMemoryBank(
    associative_memory.AssociativeMemory(embedder)
)

# Let's create an agent with the above components.
agent = entity_agent.EntityAgent(
    "Alice",
    act_component=SimpleActing(model),
    context_components={
        "observation": Observe(),
        "relevant_memories": RelevantMemories(),
        "recent_memories": RecentMemoriesImproved(),
        "memory": memory_component.MemoryComponent(raw_memory),
    },
)


# Avi comments: It took me a while to understand why ActionSpecIgnored class is needed. It seems to be a post-hoc workaround for the fact that by getting the component you might call a phase method with a differnet action spec then the one used internally by EntityAgent.

#  A better approach would be for EntityAgent to retune the value of a phase directly ensuring internally that it uses the same action spec.

# i.e., instead of self.get_entity().get_component('name').anything...

# Could have done self.get_entity().get_component_phase('pre-act', 'component-name')
# of course you can only access phases equal to or earlier than the current phase.
