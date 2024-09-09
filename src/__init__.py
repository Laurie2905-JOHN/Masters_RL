from models.envs.env import *
# Register the environment
from gymnasium.envs.registration import register


register(
    id='SchoolMealSelection-v0',
    entry_point='models.envs.env:SchoolMealSelectionContinuous',
    max_episode_steps=100
)

register(
    id='SchoolMealSelection-v1',
    entry_point='models.envs.env:SchoolMealSelectionDiscrete',
    max_episode_steps=100
)

register(
    id='SchoolMealSelection-v2',
    entry_point='models.envs.env:SchoolMealSelectionDiscreteDone',
    max_episode_steps=1000
)