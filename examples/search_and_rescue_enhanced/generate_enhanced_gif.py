# Install (once, from repo root):
#   pip install -e .
#
# Run (from repo root):
#   python examples/search_and_rescue_enhanced/generate_enhanced_gif.py
#
# Output: examples/search_and_rescue_enhanced/search_and_rescue_enhanced.gif

import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import numpy as np

from jumanji.environments.swarms.search_and_rescue.env import SearchAndRescue
from jumanji.environments.swarms.search_and_rescue.enhanced_viewer import (
    EnhancedSearchAndRescueViewer,
)
from jumanji.environments.swarms.search_and_rescue.observations import (
    AgentAndTargetObservationFn,
)

# ── configuration ──────────────────────────────────────────────────────────────
NUM_STEPS = 200
INTERVAL_MS = 100  # ms per frame in the GIF
SEED = 42

# Output path: same directory as this script
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(_HERE, "search_and_rescue_enhanced.gif")

# ── build environment ──────────────────────────────────────────────────────────
env = SearchAndRescue(
    time_limit=NUM_STEPS,
    observation=AgentAndTargetObservationFn(
        num_vision=128,
        searcher_vision_range=0.4,
        target_vision_range=0.1,
        view_angle=0.4,
        agent_radius=0.01,  # visual radius = half of target_contact_range (0.02)
        env_size=1.0,
    ),
)

# ── rollout ────────────────────────────────────────────────────────────────────
key = jax.random.PRNGKey(SEED)
state, timestep = jax.jit(env.reset)(key)

states = [state]
observations = [timestep.observation]
actions_list: list = []
rewards_list: list = []

print(f"Rolling out {NUM_STEPS} steps with random actions…")
for i in range(NUM_STEPS - 1):
    key, action_key = jax.random.split(key)
    action = jax.random.uniform(
        action_key,
        shape=(env.generator.num_searchers, 2),
        minval=-1.0,
        maxval=1.0,
    )
    state, timestep = jax.jit(env.step)(state, action)
    states.append(state)
    observations.append(timestep.observation)
    actions_list.append(np.array(action))
    rewards_list.append(np.array(timestep.reward))

# Prepend a zero action/reward for the initial frame
zero_action = np.zeros((env.generator.num_searchers, 2))
zero_reward = np.zeros(env.generator.num_searchers)
actions_list.insert(0, zero_action)
rewards_list.insert(0, zero_reward)

# ── build viewer ───────────────────────────────────────────────────────────────
print("Building enhanced viewer…")
viewer = EnhancedSearchAndRescueViewer(
    env_size=(env.generator.env_size, env.generator.env_size),
    view_angle=env._observation_fn.view_angle,
    searcher_vision_range=env._observation_fn.searcher_vision_range,
    target_vision_range=env._observation_fn.target_vision_range,
    target_contact_range=env.target_contact_range,
    target_visual_radius=env._observation_fn.agent_radius,
    max_rotate=env.searcher_params.max_rotate,
    trail_length=20,
)

# ── render ─────────────────────────────────────────────────────────────────────
print(f"Rendering animation ({NUM_STEPS} frames) — this may take a minute…")
anim = viewer.animate(
    states=states,
    observations=observations,
    actions=actions_list,
    rewards=rewards_list,
    interval=INTERVAL_MS,
    save_path=OUTPUT_PATH,
)
print(f"Done!  GIF saved to: {OUTPUT_PATH}")
