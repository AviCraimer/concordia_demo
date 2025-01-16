# see https://console.anthropic.com/workbench/f24f7f0c-6ab0-45f8-a0fd-b350cac41dd7

from typing import Dict, Literal, Mapping, Type, TypeVar, Union, TypedDict, Dict
from dataclasses import dataclass

import collections

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

# Type for components
C = TypeVar("C")


class DependentContextComponent(entity_component.ContextComponent):
    def __init__(
        self, peer_context_types: Dict[str, Type[entity_component.ContextComponent]]
    ):
        super().__init__()
        self.peer_context_types = peer_context_types


class SafeEntityAgent(entity_agent.EntityAgent):
    def __init__(
        self,
        name: str,
        act_component: entity_component.ActingComponent,
        context_components: Mapping[str, entity_component.ContextComponent],
    ):

        # Check dependencies
        for component_name, component in context_components.items():
            if isinstance(component, DependentContextComponent):
                for peer_name, peer_type in component.peer_context_types.items():
                    if peer_name not in context_components:
                        raise ValueError(
                            f"Component '{component_name}' requires peer component "
                            f"'{peer_name}' of type {peer_type.__name__}"
                        )
                    if not isinstance(context_components[peer_name], peer_type):
                        raise ValueError(
                            f"Component '{component_name}' requires peer component "
                            f"'{peer_name}' to be of type {peer_type.__name__}, but got "
                            f"{type(context_components[peer_name]).__name__}"
                        )

        # If all checks pass, create the EntityAgent
        # Note: We left out the context_processor param, this could be added later
        self.agent = entity_agent.EntityAgent(
            agent_name=name,
            act_component=act_component,
            context_components=context_components,
        )


# Usage example:
class Observe(DependentContextComponent):
    def __init__(self):
        super().__init__({"memory": memory_component.MemoryComponent})

    def pre_observe(self, observation: str) -> str:
        self.get_entity().get_component(
            "memory", type_=memory_component.MemoryComponent
        ).add(observation, {})
        return ""


class RecentMemories(DependentContextComponent):
    def __init__(self):
        super().__init__({"memory": memory_component.MemoryComponent})

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

# This should not raise an error
agent = SafeEntityAgent(
    "Alice",
    act_component=SimpleActing(model),
    context_components={
        "observation": Observe(),
        "recent_memories": RecentMemories(),
        "memory": memory_component.MemoryComponent(raw_memory),
    },
)


# This would raise an error because 'memory' component is missing
agent = SafeEntityAgent(
    "Alice",
    act_component=SimpleActing(model),
    context_components={
        "observation": Observe(),
        "recent_memories": RecentMemories(),
    },
)

# Raises: ValueError: Component 'observation' requires peer component 'memory' of type MemoryComponent


from enum import Enum, auto
from typing import Dict, Set, Tuple, Optional
import inspect
import threading
from dataclasses import dataclass

from concordia.agents import entity_agent
from concordia.typing import entity_component
from concordia.typing import entity


class Phase(Enum):
    """Phases of agent operation, in temporal order."""

    INIT = auto()
    PRE_ACT = auto()
    POST_ACT = auto()
    PRE_OBSERVE = auto()
    POST_OBSERVE = auto()
    UPDATE = auto()

    def __lt__(self, other: "Phase") -> bool:
        """Enable direct comparison of phases."""
        return self.value < other.value

    def __le__(self, other: "Phase") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "Phase") -> bool:
        return self.value > other.value

    def __ge__(self, other: "Phase") -> bool:
        return self.value >= other.value


# Define valid phase strings
PhaseString = Literal[
    "init", "pre-act", "post-act", "pre-observe", "post-observe", "update"
]

# Update mapping to include all phases
METHOD_TO_PHASE = {
    "init": Phase.INIT,
    "pre-act": Phase.PRE_ACT,
    "post-act": Phase.POST_ACT,
    "pre-observe": Phase.PRE_OBSERVE,
    "post-observe": Phase.POST_OBSERVE,
    "update": Phase.UPDATE,
}


# The main enhancement over the base EntityAgent is the ability to safely access component values from different phases while preventing problematic access patterns.
class SaferEntityAgent(entity_agent.EntityAgent):
    """
    An EntityAgent with enhanced safety features for component interactions.

    This implementation adds:
    1. Phase-aware access control between components
    2. Detection of circular dependencies
    3. Protection against accessing future phase values
    4. Special handling of the UPDATE phase
    """

    def __init__(
        self,
        agent_name: str,
        act_component: entity_component.ActingComponent,
        context_components: Dict[str, entity_component.ContextComponent],
        context_processor: Optional[entity_component.ContextProcessorComponent] = None,
    ):
        super().__init__(
            agent_name=agent_name,
            act_component=act_component,
            context_components=context_components,
            context_processor=context_processor,
        )
        # Track who calls whom in each phase
        self._phase_calls: Dict[Tuple[Phase, str], Set[str]] = {}
        # Cache for phase values
        self._phase_values: Dict[Tuple[str, Phase], str] = {}

    def _get_calling_context(self) -> Tuple[str, str]:  # (component_name, method_name)
        """
        Determine which component and method is calling get_component_phase.

        Returns:
            Tuple of (component_name, method_name)

        Raises:
            ValueError if calling context cannot be determined
        """
        frame = inspect.currentframe()
        try:
            while frame:
                local_vars = frame.f_locals
                if "self" in local_vars and isinstance(
                    local_vars["self"], entity_component.ContextComponent
                ):
                    # Get the component name
                    component_name = None
                    for name, component in self._context_components.items():
                        if component is local_vars["self"]:
                            component_name = name

                    if component_name:
                        # Get the method name
                        method_name = frame.f_code.co_name
                        return component_name, method_name

                frame = frame.f_back
            raise ValueError("Could not determine calling component and method")
        finally:
            del frame

    def get_component_phase(
        self,
        target_component: str,
        phase: Union[Phase, PhaseString],
    ) -> str:
        """
        Safely access a component's value for a specific phase.

        Args:
            target_component: Name of the component whose value is being requested
            target: Phase of the value being requested as string or Phase enum value

        Returns:
            The requested phase value

        Raises:
            ValueError for:
                - Accessing future phases
                - Accessing UPDATE phase of other components
                - Circular dependencies within the same phase
        """
        # Convert phase argument to Phase value
        target_phase: Phase = (
            METHOD_TO_PHASE[phase] if isinstance(phase, str) else phase
        )

        calling_component, calling_method = self._get_calling_context()
        calling_phase = METHOD_TO_PHASE[calling_method]

        # Check phase ordering
        if target_phase > calling_phase:
            raise ValueError(
                f"Component {calling_component} in phase {calling_phase} cannot access "
                f"future phase {target_phase} of component {target_component}"
            )

        # Special case for UPDATE phase
        if target_phase == Phase.UPDATE:
            raise ValueError(
                f"Components cannot access UPDATE phase of other components. "
                f"{calling_component} attempted to access UPDATE phase of {target_component}"
            )

        # Check for circular dependencies (only within same phase)
        if calling_phase == target_phase:
            target_key = (target_phase, target_component)
            if (
                target_key in self._phase_calls
                and calling_component in self._phase_calls[target_key]
            ):
                raise ValueError(
                    f"Circular dependency detected: {calling_component} and {target_component} "
                    f"are mutually dependent in phase {calling_phase}"
                )

            # Record this call
            caller_key = (calling_phase, calling_component)
            if caller_key not in self._phase_calls:
                self._phase_calls[caller_key] = set()
            self._phase_calls[caller_key].add(target_component)

        return self._get_phase_value(target_phase, target_component)

    def _get_phase_value(self, phase: Phase, component_name: str) -> str:
        """Retrieve the cached value for a component's phase."""
        key = (component_name, phase)
        if key not in self._phase_values:
            raise ValueError(
                f"No value available for component {component_name} in phase {phase}"
            )
        return self._phase_values[key]

    def _set_phase(self, phase: Phase) -> None:
        """Override to clear phase tracking when changing phases."""
        super()._set_phase(phase)
        self._phase_calls.clear()
        if phase == Phase.READY:
            self._phase_values.clear()

    def _cache_phase_value(self, component_name: str, phase: Phase, value: str) -> None:
        """Cache a value for a component's phase."""
        self._phase_values[(component_name, phase)] = value

    def _parallel_call_(
        self,
        method_name: str,
        *args,
    ) -> entity_component.ComponentContextMapping:
        """Override to cache phase values from component calls."""
        results = super()._parallel_call_(method_name, *args)
        phase = METHOD_TO_PHASE.get(method_name)
        if phase:
            for component_name, value in results.items():
                self._cache_phase_value(component_name, phase, value)
        return results
