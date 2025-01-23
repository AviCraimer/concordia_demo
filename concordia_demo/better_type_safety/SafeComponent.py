from typing import (
    Any,
    Dict,
    FrozenSet,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    Dict,
    Generic,
    TypedDict,
    overload,
    cast,
)
import types
from types import MappingProxyType
from typing_extensions import override
from concordia.typing import entity
import functools

from concordia.agents import entity_agent
from concordia.associative_memory import associative_memory
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.typing import entity as entity_lib
from concordia.components.agent import no_op_context_processor
import threading
from concordia.utils import concurrency


ComponentResults: TypeAlias = Mapping[str, str]


class PrevPhasesReady(TypedDict):
    ready: ComponentResults


class PrevPhasesPre(TypedDict):
    ready: ComponentResults
    pre: ComponentResults


class PrevPhasesPost(TypedDict):
    ready: ComponentResults
    pre: ComponentResults
    post: ComponentResults


PrevPhaseResults: TypeAlias = Union[PrevPhasesReady, PrevPhasesPre, PrevPhasesPost]


PhaseMethodName: TypeAlias = Literal[
    "ready_act",
    "pre_act",
    "post_act",
    "update_act",
    "ready_observe",
    "pre_observe",
    "post_observe",
    "update_observe",
]

PhasePrefix: TypeAlias = Literal["ready", "pre", "post", "update"]


def phase_method_to_prefix(method_name: PhaseMethodName) -> PhasePrefix:
    return method_name.split("_")[0]  # type: ignore  # We know this will be a PhasePrefix


class SafeContextComponent(entity_component.BaseComponent):

    def __init__(self, name: str, peer_component_names: frozenset[str]):
        super().__init__()
        self.name = name
        self.peer_component_names = peer_component_names

    def get_peer_value(self, values: dict[str, str], peer_name: str) -> str:
        """Safely access a peer component's value.

        Args:
            values: Dictionary of component values
            peer_name: Name of the peer component whose value we want

        Returns:
            The value for the peer component

        Raises:
            ValueError: If peer_name is not in peer_component_names
            KeyError: If peer_name is not in values (Python built-in error)
        """
        if peer_name not in self.peer_component_names:
            raise ValueError(
                f"Component '{self.name}' attempted to access value of '{peer_name}' "
                f"which is not in its declared peer_component_names {self.peer_component_names}"
            )
        return values[peer_name]

    def ready_act(self) -> str:
        return ""

    def pre_act(
        self, action_spec: entity_lib.ActionSpec, prev_phases: PrevPhasesReady
    ) -> str:
        del action_spec
        return ""

    def post_act(self, action_attempt: str, prev_phases: PrevPhasesPre) -> str:
        return ""

    def update_act(self, prev_phases: PrevPhasesPost) -> None:
        return None

    def ready_observe(
        self,
    ) -> str:
        return ""

    def pre_observe(self, observation: str, prev_phases: PrevPhasesReady) -> str:
        del observation
        return ""

    def post_observe(self, prev_phases: PrevPhasesPre) -> str:
        return ""

    def update_observe(self, prev_phases: PrevPhasesPost) -> None:
        return None


# Used in SafeEntityAgent for self._phase_results
class PhaseResults(TypedDict):
    ready: Optional[ComponentResults]
    pre: Optional[ComponentResults]
    post: Optional[ComponentResults]


initial_phase_results: PhaseResults = {
    "ready": None,
    "pre": None,
    "post": None,
}


class SafeEntityAgent(entity_component.EntityWithComponents):
    def __init__(
        self,
        agent_name: str,
        act_component: entity_component.ActingComponent,
        context_components: list[SafeContextComponent],
    ):
        super().__init__()
        self._agent_name = agent_name
        self._control_lock = threading.Lock()
        self._phase_lock = threading.Lock()
        self._phase = entity_component.Phase.READY

        # Cache for phase results
        self._phase_results: PhaseResults = initial_phase_results

        self._act_component = act_component
        self._act_component.set_entity(self)

        self._context_components: Mapping[str, SafeContextComponent] = {}
        for component in context_components:
            self._context_components[component.name] = component

        # Freeze the components dictionary
        self._context_components = MappingProxyType(self._context_components)

        for component in context_components:
            self.validate_context_component(component)

    def validate_context_component(self, component: SafeContextComponent) -> None:
        missing_peers = component.peer_component_names - self._context_components.keys()
        if missing_peers:
            raise ValueError(
                f"Component '{component.name}' requires peer components {missing_peers} "
                f"which are not present in the agent"
            )

    @functools.cached_property
    def name(self) -> str:
        return self._agent_name

    def get_phase(self) -> entity_component.Phase:
        with self._phase_lock:
            return self._phase

    def _set_phase(self, phase: entity_component.Phase) -> None:
        with self._phase_lock:
            self._phase.check_successor(phase)
            self._phase = phase

    def get_component(
        self,
        name: str,
        *,
        type_: type[SafeContextComponent] = SafeContextComponent,
    ) -> SafeContextComponent:
        component: SafeContextComponent = self._context_components[name]
        if not component:
            raise ValueError(f"Component {name} not found in get_component")
        if not isinstance(component, SafeContextComponent):
            raise TypeError(f"Component {name} is not a SafeContextComponent")
        return component

    def get_prev_phases(self, method_name: PhaseMethodName) -> PrevPhaseResults:
        prefix = phase_method_to_prefix(method_name)

        if prefix == "pre":
            if self._phase_results["ready"] is None:
                raise ValueError(f"Missing ready results for {method_name}")
            return {"ready": self._phase_results["ready"]}

        elif prefix == "post":
            if (
                self._phase_results["ready"] is None
                or self._phase_results["pre"] is None
            ):
                raise ValueError(f"Missing ready or pre results for {method_name}")
            return {
                "ready": self._phase_results["ready"],
                "pre": self._phase_results["pre"],
            }

        elif prefix == "update":
            if (
                self._phase_results["ready"] is None
                or self._phase_results["pre"] is None
                or self._phase_results["post"] is None
            ):
                raise ValueError(f"Missing phase results for {method_name}")
            return {
                "ready": self._phase_results["ready"],
                "pre": self._phase_results["pre"],
                "post": self._phase_results["post"],
            }

        raise ValueError(f"Invalid method name: {method_name}")

    def _parallel_call_(
        self,
        method_name: PhaseMethodName,
        *args,
    ) -> None:
        tasks = {
            name: functools.partial(
                getattr(component, method_name),
                *args,
                prev_phases=self.get_prev_phases(method_name),
            )
            for name, component in self._context_components.items()
        }
        results: Mapping[str, str] = concurrency.run_tasks(tasks)
        prefix = phase_method_to_prefix(method_name)
        # Update phase results (except for update phases which return None)
        if not prefix == "update":
            self._phase_results[prefix] = results

    def act(self, action_spec: entity.ActionSpec = entity.DEFAULT_ACTION_SPEC) -> str:
        with self._control_lock:
            # Ready phase
            if self._phase != entity_component.Phase.READY:
                raise ValueError("act has been called when phase is not set to READY")

            # Clear any previous phase results
            self._phase_results = initial_phase_results
            self._parallel_call_("ready_act")

            # Pre-act phase
            self._set_phase(entity_component.Phase.PRE_ACT)
            self._parallel_call_("pre_act", action_spec)
            if not self._phase_results["pre"]:
                raise ValueError(
                    "Pre-act results not set (this should never happen at this stage)"
                )
            action_attempt = self._act_component.get_action_attempt(
                self._phase_results["pre"], action_spec
            )

            # Post-act phase
            self._set_phase(entity_component.Phase.POST_ACT)
            self._parallel_call_("post_act", action_attempt)

            # Update phase
            self._set_phase(entity_component.Phase.UPDATE)
            self._parallel_call_("update_act")

            # Back to ready
            self._set_phase(entity_component.Phase.READY)
            self._phase_results = initial_phase_results
            return action_attempt

    def observe(self, observation: str) -> None:
        with self._control_lock:
            # Ready phase
            if self._phase != entity_component.Phase.READY:
                raise ValueError(
                    "observe has been called when phase is not set to READY"
                )
            # Clear any previous phase results
            self._phase_results = initial_phase_results
            self._parallel_call_("ready_observe")

            # Pre-observe phase
            self._set_phase(entity_component.Phase.PRE_OBSERVE)
            self._parallel_call_("pre_observe", observation)

            # Post-observe phase
            self._set_phase(entity_component.Phase.POST_OBSERVE)
            self._parallel_call_("post_observe")

            # Update phase
            self._set_phase(entity_component.Phase.UPDATE)
            self._parallel_call_("update_observe")

            # Back to ready
            self._set_phase(entity_component.Phase.READY)
            self._phase_results = initial_phase_results


"""
SafeEntityAgent and SafeContextComponent provide a safer implementation of Concordia's entity system
with explicit phase dependencies and stricter access controls.

Phase Flow:
-----------
Each action or observation cycle goes through the following phases:
    READY -> PRE_ACT -> POST_ACT -> UPDATE -> READY
    - Between pre-act and post-act, the acting agent is called to produce an action attempt.
    READY -> PRE_OBSERVE -> POST_OBSERVE -> UPDATE -> READY


In each phase, components can access results from all previous phases in the current cycle:
    - ready_act/observe: No previous results
    - pre_act/observe: Access to ready results
    - post_act/observe: Access to ready and pre results
    - update_act/observe: Access to ready, pre, and post results

All phase results are cleared when returning to READY state.

Action vs Observation Cycles:
---------------------------
The agent alternates between action and observation cycles:

Action Cycle:
1. Components provide context for action decision (pre_act)
2. ActingComponent generates action attempt based on the results of pre-act and the action spec.
3. Components can record and respond to the attempted action in post_act. Different components may care about different things so this is a chance to re-construe the action attempt in a way relevant to each component individually.
4. Components update their internal state based on the post_act output. e.g., a memory may update itself to record the action attempt, a will-power component may subtract points from it's will-power score etc.

Note: post_act and update_act only know what action was attempted, not its outcome.
Actual outcomes can be received as observations in a subsequent observation cycle.

Observation Cycle:
1. ready_observe - This is the starting state of the components before they have recieved the new observation.
2. Components process new observation (pre_observe)
3. Components finalize observation processing (post_observe)
 - Unlike with the act cycle, there is no observation attempt generated, but the distinction betweeen pre and post observe is still important since in post_observe, each component has access to the observation response of other components.
4. Components update internal state (update_observe)

Implementing Safe Components:
---------------------------
To implement a SafeContextComponent:

1. Define phase methods with appropriate access to previous results:

    def pre_act(
        self,
        action_spec: ActionSpec,
        prev_phases: PrevPhasesPre  # Contains only 'ready' results
    ) -> str:
        # Access previous results via prev_phases['ready'][component_name]
        return "context for action decision"

    def post_act(
        self,
        action_attempt: str,
        prev_phases: PrevPhasesPost  # Contains 'ready' and 'pre' results
    ) -> str:
        # Can access both ready and pre_act results
        return "response to attempted action"

2. All phase methods except update_act/observe should return a string that will be
   available to other components in subsequent phases.

3. Update methods are for modifying internal state based on the complete cycle:

    def update_act(
        self,
        prev_phases: PrevPhasesUpdate  # Contains 'ready', 'pre', and 'post' results
    ) -> None:
        # Update internal state based on action cycle
        pass

Safety Features:
--------------
- Components cannot access future phase results
- Components cannot access UPDATE phase results from other components
- Phase results are properly cleared between cycles
- Explicit phase dependencies through prev_phases parameter
- Type-safe access to component results

Example Usage:
-------------
class MemoryComponent(SafeContextComponent):
    def pre_act(
        self,
        action_spec: ActionSpec,
        prev_phases: PrevPhasesPre
    ) -> str:
        # Access ready phase results from other components
        location = prev_phases['ready']['location_component']
        return f"Agent remembers being at {location}"

    def post_act(
        self,
        action_attempt: str,
        prev_phases: PrevPhasesPost
    ) -> str:
        # Access both ready and pre results
        location = prev_phases['ready']['location_component']
        memory = prev_phases['pre']['memory_component']
        return f"Recording attempt to {action_attempt} at {location}"

    def update_act(
        self,
        prev_phases: PrevPhasesUpdate
    ) -> None:
        # Update internal memory state based on complete action cycle
        pass

Notes:
------
- Components should not try to access phase results directly through the entity
- All inter-component communication should happen through the prev_phases parameter
- Components should be designed to handle missing results from optional components
- The UPDATE phase is for internal state changes only; its results are not shared
"""
