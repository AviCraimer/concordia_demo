# Example from https://github.com/google-deepmind/concordia/blob/main/examples/brainstorm/brainstorm.ipynb

import os
import concurrent.futures
import datetime
import random
from typing import Callable
from concordia import components as generic_components
from concordia.agents import deprecated_agent as basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory.blank_memories import MemoryFactory
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components.agent import to_be_deprecated as agent_components
from concordia.components import game_master as gm_components
from concordia.document import interactive_document
from concordia.environment import game_master
from concordia.environment.scenes import conversation as conversation_scene

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
from concordia_demo.llm_integrations.claude_concordia_semantic_sim import (
    SemanticEmbedder,
)
from concordia_demo.llm_integrations.claude_concordia_model import ClaudeModel
from concordia_demo.examples.run_capture import HTMLWriter, RunPaths
from collections.abc import Sequence
from typing import Any
import dataclasses

global capture_paths
capture_paths: RunPaths

# pyright: ignore[reportUnboundVariable]
if isinstance(capture_paths, RunPaths):
    raise ValueError("capture_paths global variable not working")

# pyright: ignore[reportUnboundVariable]
html_writer = HTMLWriter(capture_paths)


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
model = ClaudeModel()
embedder = SemanticEmbedder(reference_text=PROJECT_PREMISE)

blank_memory_factory = MemoryFactory(
    model=model,
    embedder=embedder,
    importance=importance_model.importance,
    clock_now=clock.now,
)


def test_claude_concordia_integration():
    """Test the integration of Claude with Concordia components."""

    print("Testing Claude-Concordia Integration...")

    # 1. Test ClaudeLanguageModel
    print("\n1. Testing ClaudeLanguageModel...")
    claude_model = ClaudeModel()
    completion = claude_model.sample_text("What is 2+2? Answer with just the number.")
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
        importance=importance_model.importance,
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class AgentConfig:
    """A card that describes a player.

    Attributes:
      name: name of the agent.
      gender: the gender of the agent.
      traits: any traits to use while generating formative memories. For example,
        big five.
      context: agent formative memories will be generated with this context
      specific_memories: inject these specific memories. Split memories at newline
        characters. Can be left blank if not used.
      goal: defines agents goal. Can be left blank if not used.
      date_of_birth: the date of birth for the agent.
      formative_ages: ages at which the formative episodes will be created
      formative_memory_importance: the importance value of formative memories.
      extras: a field for the user to keep any experiment specific data they need
        to define an agent
    """

    name: str
    gender: str
    traits: str
    selected_works: Sequence[str]
    goal: str = ""
    biography: str = ""
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)


def specific_memories_from_selected_works(player_config: AgentConfig) -> list[str]:
    """Create memories per player using their role as moderator or debater."""
    specific_memories = []
    player_name = player_config.name
    for work in player_config.selected_works:
        specific_memories += [(f'[writing] of {player_name}: "{work}"')]
        idea = model.sample_text(
            (
                f'Consider the paper summarised here: "{work}". Without repeating '
                "the title, a two sentence TLDR of its most important and distinctive "
                "idea is that "
            ),
            terminators=("\n",),
        )
        specific_memories += [(f"[idea] of {player_name}: {idea}")]
        print(f"idea: {idea}.")

    return specific_memories


def cat_with_dropout(inputs: list[str], dropout_rate: float = 0.2) -> str:
    result = "\n".join([x for x in inputs if random.random() > dropout_rate])
    return result


class MindStream(agent_components.observation.Observation):

    def state(self):
        mems = self._memory.retrieve_time_interval(
            self._clock_now() - self._timeframe, self._clock_now(), add_time=True
        )

        if self._verbose:
            self._log("\n".join(mems) + "\n")
        return "\n".join(mems) + "\n"


def build_agent(
    agent_config,
    unused_player_names: list[str],
    unused_measurements: measurements_lib.Measurements | None = None,
):

    agent_name = agent_config.name
    mem = blank_memory_factory.make_blank_memory()

    memories_from_work = specific_memories_from_selected_works(agent_config)

    for item in memories_from_work:
        if item:
            mem.add(item, importance=1.0)

    bio = generic_components.constant.ConstantComponent(
        state=agent_config.biography, name="biography"
    )
    time = generic_components.report_function.ReportFunction(
        name="Current time",
        function=clock.current_time_interval_str,
    )
    traits = generic_components.constant.ConstantComponent(
        state=agent_config.traits, name="psychological traits"
    )
    current_obs = MindStream(
        agent_name=agent_name,
        clock_now=clock.now,
        memory=mem,
        timeframe=clock.get_step_size() * 2,
        component_name="current observations",
    )

    convo_so_far = generic_components.report_function.ReportFunction(
        name="memory of the conversation",
        function=lambda: cat_with_dropout(list(mem.retrieve_by_regex(' -- "'))),
    )
    ideas = generic_components.report_function.ReportFunction(
        name="ideas",
        function=lambda: cat_with_dropout(list(mem.retrieve_by_regex("idea"))),
    )
    writing = generic_components.report_function.ReportFunction(
        name="writings",
        function=lambda: cat_with_dropout(list(mem.retrieve_by_regex("writing"))),
    )

    # Setup the reflection component and its related components.
    topic_of_debate = generic_components.report_function.ReportFunction(
        name="Topic of the workshop", function=lambda: PROJECT_PREMISE
    )
    goal_of_debate = generic_components.report_function.ReportFunction(
        name="Goal of the workshop", function=current_goal
    )
    # The agent's subpersonal intuition contains a bias toward thinking that
    # they themselves are the best.

    reflection = agent_components.creative_reflection.CreativeReflection(
        name="reflection",
        model=model,
        memory=mem,
        agent_name=agent_name,
        source_of_abstraction=[convo_so_far, writing, ideas],
        topic_component=topic_of_debate,
        clock_now=clock.now,
        verbose=False,
    )
    all_components = [
        bio,
        traits,
        topic_of_debate,
        goal_of_debate,
        reflection,
        current_obs,
    ]
    agent = basic_agent.BasicAgent(
        model,
        agent_name=agent_name,
        clock=clock,
        components=[time] + all_components,
        update_interval=UPDATE_INTERVAL,
    )

    return agent, mem


# SECTION: Configure and build the agents

# @title agent configs
TRAIT_LEVELS = ["low", "medium", "high"]


def make_random_big_five() -> str:
    return str(
        {
            "extraversion": np.random.choice(TRAIT_LEVELS),
            "neuroticism": np.random.choice(TRAIT_LEVELS),
            "openness": np.random.choice(TRAIT_LEVELS),
            "conscientiousness": np.random.choice(TRAIT_LEVELS),
            "agreeableness": np.random.choice(TRAIT_LEVELS),
        }
    )


# The people and writings below were generated by asking an LLM to talk about
# alchemy. The LLM was told it was writing fiction, so the cited works are
# probably not real, though some of them might be.
player_configs = [
    AgentConfig(
        name="Robert Fludd",
        gender="male",
        traits=make_random_big_five(),
        biography=(
            "Robert Fludd, a 17th-century English physician and prolific "
            "author, was a prominent figure in the world of esoteric "
            "philosophy and Rosicrucianism. He passionately defended "
            "Paracelsian medicine, which emphasized the interconnectedness of "
            "the human body, the cosmos, and the divine. Fludd's works, "
            "filled with intricate diagrams and symbolic imagery, explored "
            "diverse topics such as cosmology, astrology, alchemy, and the "
            "occult. He engaged in a heated debate with Johannes Kepler, "
            "defending a hermetic approach to knowledge against Kepler's "
            "scientific rationalism. Fludd's writings, though controversial "
            "and often criticized for their obscurity, continue to intrigue "
            "scholars interested in the intersection of science, religion, and "
            "the occult during the Renaissance period."
        ),
        selected_works=[
            (
                "Utriusque Cosmi Historia (The History of the Two Worlds): A "
                "magnum opus that explores the macrocosm (the universe) and "
                "microcosm (man). Fludd illustrates the interconnectedness of "
                "all things through detailed diagrams of the Great Chain of "
                "Being and the cosmic harmony of music. Alchemically, this "
                "work emphasizes the importance of understanding the celestial "
                "influences on terrestrial matter, a key principle in "
                "manipulating the Prima Materia."
            ),
            (
                "Tractatus Apologeticus (Apologetic Treatise): A defense of "
                "Rosicrucianism and Paracelsian medicine, Fludd argues for the "
                "integration of spiritual and material aspects in healing. "
                "He describes the alchemical process of purification and the "
                "role of the alchemist as a facilitator of nature's "
                "transformative power, using the language of the "
                "'Universal Spirit' and the 'World Soul.'"
            ),
            (
                "Philosophia Moysaica (Mosaic Philosophy): Fludd interprets "
                "the biblical creation story through an alchemical lens, "
                "equating the act of creation with the process of separation "
                "and coagulation. He discusses the 'divine fire' that animates "
                "matter and the 'waters of the abyss' from which all things "
                "emerge, echoing the alchemical concepts of Solve et Coagula."
            ),
            (
                "Summum Bonum (The Supreme Good): Fludd delves into the "
                "mystical aspects of alchemy, discussing the concept of the "
                "'divine light' that illuminates the path to spiritual "
                "enlightenment. He describes the alchemical process as a "
                "journey of purification and transformation, culminating in "
                "the attainment of the 'Philosopher's Stone,' a symbol of "
                "spiritual perfection."
            ),
            (
                "Integrum Morborum Mysterium (The Complete Mystery of "
                "Diseases): Fludd applies alchemical principles to medicine, "
                "emphasizing the importance of balancing the three primary "
                "elements (sulfur, mercury, and salt) in the human body. He "
                "advocates for using spagyric remedies, which are prepared "
                "through alchemical processes to enhance their healing "
                "properties, thereby restoring harmony and balance to the "
                "patient's internal microcosm."
            ),
        ],
        extras={},
    ),
    AgentConfig(
        name="Paracelsus",
        gender="male",
        traits=make_random_big_five(),
        biography=(
            "Born Theophrastus von Hohenheim in 1493, Paracelsus was a Swiss "
            "physician, alchemist, and revolutionary thinker who challenged "
            "the medical establishment of his time. He rejected traditional "
            "Galenic medicine, advocating for a holistic approach that "
            "combined observation, experience, and chemical remedies. "
            "Paracelsus emphasized the importance of understanding the "
            "chemical properties of substances to treat diseases, laying the "
            "groundwork for modern pharmacology. His controversial ideas and "
            "flamboyant personality earned him both admirers and detractors, "
            "but his contributions to medicine and chemistry remain "
            "significant. Paracelsus died in 1541, leaving behind a legacy of "
            "innovation and a profound impact on the development of medical "
            "science."
        ),
        selected_works=[
            (
                "Coelum Philosophorum (The Heaven of Philosophers): A "
                "foundational alchemical text where Paracelsus outlines his "
                "theory of the Tria Prima, the three fundamental principles of "
                "salt, sulfur, and mercury, which he believed formed the basis "
                "of all matter. This work revolutionized alchemical thought "
                "and laid the groundwork for his unique approach to medicine."
            ),
            (
                "Archidoxis Magica (The Archidoxes of Magic): A collection of "
                "ten books exploring the relationship between medicine, "
                "alchemy, and astrology. Paracelsus delves into the occult "
                "properties of plants, minerals, and celestial bodies, "
                "demonstrating how these can be harnessed for healing and "
                "spiritual transformation."
            ),
            (
                "De Natura Rerum (On the Nature of Things): An exploration of "
                "the natural world, where Paracelsus discusses the origins of "
                "diseases, the importance of understanding the chemical "
                "properties of substances, and the role of the physician as a "
                "healer. This work reveals his innovative approach to "
                "medicine, which combined alchemical principles with empirical "
                "observation."
            ),
            (
                "Liber Paragranum (The Book of Paragranum): A comprehensive "
                "treatise on medical theory and practice. Paracelsus "
                "challenges traditional Galenic medicine, advocating for a "
                "more holistic approach that takes into account the patient's "
                "individual constitution and environment. He emphasizes the "
                "importance of experience and experimentation in medicine, "
                "laying the groundwork for modern medical research."
            ),
            (
                "Opus Paramirum (The Book of Supreme Wonders): A collection of "
                "writings on various topics, including alchemy, medicine, "
                "astrology, and theology. Paracelsus presents his unique "
                "vision of the universe and humanity's place within it, "
                "offering insights into his philosophy and worldview. Part "
                "One: Foundations establishes Paracelsus' fundamental "
                "principles, including his theory of the three primes (salt, "
                "sulfur, and mercury) and the five entia (invisible forces "
                "that govern the world). He challenges the prevailing "
                "Galenic humoral theory, arguing for a chemical understanding "
                "of the body and its ailments. Part Two: Etiology explores "
                "the origins and causes of diseases. Paracelsus introduces "
                "the concept of ens astrale, a celestial influence that can "
                "affect both physical and mental health. And, lastly a part "
                "on Philosophy and Practice summarizes Paracelsus' philosophy "
                "of medicine, emphasizing the interconnectedness of the "
                "microcosm (the human body) and the macrocosm (the universe)."
            ),
        ],
        extras={},
    ),
    AgentConfig(
        name="Isaac Newton",
        gender="male",
        traits=make_random_big_five(),
        biography=(
            "Born prematurely on Christmas Day in 1642, Isaac Newton defied "
            "expectations by becoming one of history's most influential "
            "scientists. His insatiable curiosity led him to unravel the "
            "mysteries of gravity, laying the foundation for our understanding "
            "of the universe.  Not content with just one field, Newton's genius "
            "extended to mathematics, where he co-invented calculus, a tool "
            "still essential in modern science and engineering.  His "
            "groundbreaking work in optics revealed the true nature of light "
            "and color, forever changing how we perceive the world around us. "
            "A true polymath, Newton's interests spanned alchemy, theology, "
            "and even economics, making him a Renaissance man centuries ahead "
            "of his time.  Despite his monumental achievements, Newton "
            "remained a complex and often solitary figure, driven by an "
            "unyielding passion for knowledge that continues to inspire "
            "scientists and thinkers today."
        ),
        selected_works=[
            (
                "The Principia. In 1687, Newton unleashed his magnum opus, "
                "often simply called the Principia. It is a cornerstone of "
                "scientific literature, boldly presenting Newton's three laws "
                "of motion, which elegantly describe the relationship between "
                "a body and the forces acting upon it. This groundbreaking "
                "work also unveiled Newton's law of universal gravitation, a "
                "fundamental principle that governs the attraction between any "
                "two objects with mass in the universe. Through rigorous "
                "mathematical proofs and insightful observations, Newton "
                "demonstrated how these laws could explain a wide range of "
                "phenomena, from the elliptical orbits of planets to the "
                "rhythmic ebb and flow of tides. The Principia not only "
                "provided a comprehensive framework for understanding the "
                "physical world, but also established a new standard for "
                "scientific inquiry, inspiring generations of scientists to "
                "explore the mysteries of nature with mathematical rigor and "
                "empirical evidence."
            ),
            (
                "Newton's alchemical manuscripts, hidden from public view for "
                "centuries, were concerned with a wide range of topics, from "
                "transmuting base metals into gold, a pursuit known as "
                "chrysopoeia, to deciphering the secrets of the elusive "
                "philosopher's stone. Newton meticulously recorded his "
                "experiments, often repeating them multiple times with "
                "slight variations, meticulously documenting his observations "
                "and interpretations. His writings reveal a deep "
                "understanding of alchemical symbolism and a belief in the "
                "interconnectedness of matter and spirit."
            ),
            (
                "Newton's most famous article is the final chapter of The "
                "Principia, which was lost and only just rediscovered in 2024. "
                "It is titled 'Addendum: Of the Celestial Chymistry'. The "
                "full text is as follows: "
                "In this, my final discourse, I posit a most curious "
                "connection between the celestial mechanics laid forth in this "
                "tome and the ancient art of chymistry. For as the planets "
                "dance in their elliptical orbits, bound by the invisible hand "
                "of gravity, so too do the elements dance in their "
                "transformations, bound by a yet unseen force. In the "
                "celestial sphere, the ellipse is the path of harmony, a "
                "balance between the sun's attraction and the planet's "
                "inertia. In the chymical sphere, might there not be a "
                "similar ellipse, a balance between the fiery and watery "
                "principles, the sulfur and the mercury? Consider the process "
                "of calcination, wherein a metal is heated to ash. This is "
                "akin to a planet flung far from the sun, its volatile "
                "spirits dissipating into the aether. Yet, as the planet "
                "returns to perihelion, drawn by the sun's embrace, so too can "
                "the ash be revived, its essence reconstituted through the "
                "addition of the appropriate elixir.Thus, I propose that the "
                "elliptical orbit is not merely a celestial phenomenon, but a "
                "universal principle, governing not only the motions of "
                "heavenly bodies but also the transformations of matter. "
                "Perhaps the elusive Philosopher's Stone, the key to "
                "transmutation, is not a tangible substance but a state of "
                "perfect chymical equilibrium, achieved when the elements "
                "trace their elliptical paths in harmony. This is but a "
                "hypothesis, a spark to ignite further inquiry. Yet, I believe "
                "that by uniting the celestial and chymical spheres, we may "
                "unlock the secrets of nature and achieve a deeper "
                "understanding of the cosmos. For as above, so below, and the "
                "dance of the planets may yet reveal the dance of the elements."
            ),
            (
                "Lux Naturae: The Alchemical Optics (Opticks Alchymica): In "
                "this intriguing work, Newton explores the parallels between "
                "the behavior of light and the transformative processes of "
                "alchemy. He postulates that just as light can be refracted "
                "and dispersed into a spectrum of colors, so too can matter be "
                "broken down and reconstituted into new forms. Newton "
                "investigates the alchemical significance of colors, "
                "associating each hue with specific metals and planetary "
                "influences. He also delves into the concept of the "
                "'luminiferous aether,' a hypothetical medium for the "
                "transmission of light, suggesting that it could be the key to "
                "unlocking the secrets of transmutation."
            ),
            (
                "De Mercurio Philosophorum: The Mercurial Principle (De "
                "Principio Mercuriali): This treatise focuses on the role of "
                "mercury in alchemical operations. Newton examines the various "
                "forms of mercury, from the common quicksilver to the elusive "
                "'philosophical mercury,' believed to be the essential "
                "ingredient in the creation of the Philosopher's Stone. He "
                "describes elaborate procedures for purifying and manipulating "
                "mercury, drawing connections between its volatility, "
                "fluidity, and transformative properties. Newton also "
                "explores the symbolism of mercury as the hermaphrodite, a "
                "dual-natured substance that unites opposites and facilitates "
                "the alchemical marriage."
            ),
        ],
        extras={},
    ),
]

NUM_PLAYERS = len(player_configs)
player_configs.reverse()

player_names = [player.name for player in player_configs][:NUM_PLAYERS]
measurements = measurements_lib.Measurements()

players = []
memories = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PLAYERS) as pool:
    for agent, mem in pool.map(
        build_agent,
        player_configs[:NUM_PLAYERS],
        # All players get the same `player_names`.
        [player_names] * NUM_PLAYERS,
        # All players get the same `measurements` object.
        [measurements] * NUM_PLAYERS,
    ):
        players.append(agent)
        memories[agent.name] = mem

# @title Summarise the perspective of each player
player_logs = []
player_log_names = []
for player in players:
    name = player.name
    detailed_story = "\n".join(
        memories[player.name].retrieve_recent(k=1000, add_time=True)
    )
    summary = player.state().splitlines()

    all_player_mem = memories[player.name].retrieve_recent(k=1000, add_time=True)
    all_player_mem = ["Player state:", summary, "Memories:"] + all_player_mem
    player_html = html_lib.PythonObjectToHTMLConverter(all_player_mem).convert()
    player_logs.append(player_html)
    player_log_names.append(f"{name}")

tabbed_html = html_lib.combine_html_pages(
    player_logs,
    player_log_names,
    summary="",
    title="Backstory of the players",
)

tabbed_html = html_lib.finalise_html(tabbed_html)
html_writer.add_section(tabbed_html)


# SECTION: Build GM

call_to_speech = (
    "Given the above, generate what {agent_name} would say next? Take their "
    "ideas and reflections and the goal of the workship into account. Respond "
    'in the format `{agent_name} -- "..."`'
)

clock.advance()

debate_event = (
    f"{players[0].name}, {players[1].name} and {players[2].name} are in a"
    f" workshop.\n Their current goal is {current_goal()}."
)

for player in players:
    player.observe(debate_event)

for player in players:
    player.observe("It is time to for a discussion now")


convo_scene = conversation_scene.make_conversation_game_master(
    players,
    clock=clock,
    model=model,
    memory_factory=blank_memory_factory,
    name="Brainstorm",
    premise=debate_event,
    call_to_speech=call_to_speech,
    review_participants=True,
    check_for_termination=False,
    randomise_initiative=True,
)
with clock.higher_gear():
    clock.advance()
    output = convo_scene.run_episode(10)

first_convo_html = html_lib.PythonObjectToHTMLConverter(
    convo_scene.get_history()
).convert()

html_writer.add_section(first_convo_html)

essays = []
for player in players:
    prompt = interactive_document.InteractiveDocument(model)
    prompt.statement(player.state())
    agent_name = player.name
    result = prompt.open_question(
        "Generate an essay on the topic of the workshop from the perspective of"
        f" {agent_name}. The goal of the essay is to summarise the conversation"
        f" and {current_goal}. Write in the style of {agent_name}, taking their"
        " ideas and reflections into account. Format the output as html",
        max_tokens=5000,
        terminators=(),
    )
    essays.append(result)

tabbed_html = html_lib.combine_html_pages(
    essays,
    [player.name for player in players],
    summary="",
    title="First essays by participants",
)

first_essays_html = html_lib.finalise_html(tabbed_html)
html_writer.add_section(first_essays_html)

clock.advance()

current_goal.update_subgoal(PROJECT_SUBGOALS[1])
debate_event = (
    f"{players[0].name}, {players[1].name} and {players[2].name} "
    f"are in a workshop, discussing {PROJECT_PREMISE}. Current "
    f"goal is {current_goal()}"
)

convo_scene = conversation_scene.make_conversation_game_master(
    players,
    clock=clock,
    model=model,
    memory_factory=blank_memory_factory,
    name="Debate",
    premise=debate_event,
    call_to_speech=call_to_speech,
    review_participants=True,
)
with clock.higher_gear():
    clock.advance()
    output = convo_scene.run_episode(10)

second_convo_html = html_lib.PythonObjectToHTMLConverter(
    convo_scene.get_history()
).convert()

html_writer.add_section(second_convo_html)


game_master_memory = associative_memory.AssociativeMemory(
    sentence_embedder=embedder, importance=importance_model.importance, clock=clock.now
)
primary_environment, game_master_memory = basic_game_master.build_game_master(
    model=model,
    embedder=embedder,
    importance_model=importance_model_gm,
    clock=clock,
    players=players,
    shared_memories=[f"{PROJECT_PREMISE}\n{PROJECT_CONTEXT}"],
    shared_context=f"{PROJECT_GOAL}",
    blank_memory_factory=blank_memory_factory,
    memory=game_master_memory,
)

episode_length = 4  # @param {type: 'integer'}
for _ in range(episode_length):
    primary_environment.step()

results_html = basic_game_master.create_html_log(
    model=model,
    primary_environment=primary_environment,
    secondary_environments=[],
)

html_writer.add_section(results_html)

html_writer.finalize()
