from collections import defaultdict
from typing import Dict, Any, Optional, Set, List

import gym
import numpy as np
from collections import defaultdict
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_shortest_paths

from envs.flatland.utils.gym_env import StepOutput, FlatlandGymEnv

class CustomRewardWrapper(gym.Wrapper):

    def __init__(self, env, finished_reward=1, not_finished_reward=-1, deadlock_reward=-0.5) -> None:
        super().__init__(env)
        self._finished_reward = finished_reward
        self._not_finished_reward = not_finished_reward
        self._deadlock_reward = deadlock_reward
        self._deadlocked_agents = []
        self.max_depth = 30

    def get_shortest_distance(self, rail_env: RailEnv, agent_id: int):

        agent = rail_env.agents[agent_id]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        # possible_transitions = rail_env.rail.get_transitions(*agent_virtual_position, agent.direction)
        distance_map = rail_env.distance_map
        min_distance = get_shortest_paths(distance_map=distance_map, max_depth=self.max_depth, agent_handle=agent_id)[agent_id]

        if min_distance is None:
            return self.max_depth
        else:
            return len(min_distance)

        # for movement in list(range(4)):
        #     if possible_transitions[movement]:
        #         if movement == agent.direction:
        #             action = RailEnvActions.MOVE_FORWARD
        #         elif movement == (agent.direction + 1) % 4:
        #             action = RailEnvActions.MOVE_RIGHT
        #         elif movement == (agent.direction - 1) % 4:
        #             action = RailEnvActions.MOVE_LEFT
        #         else:
        #             raise ValueError("Wtf, debug this shit.")
        #         distance = distance_map[get_new_position(agent_virtual_position, movement) + (movement,)]
        #         possible_steps.append(distance)



        # min_distance = sorted(possible_steps)[0]
        #
        # if min_distance is not None:
        #     if min_distance == 1:
        #         return min_distance * 2
        #     else:
        #         return min_distance
        # else:
        #     min_distance = 30

        # possible_steps = sorted(possible_steps, key=lambda step: step[1])
        #
        # if len(possible_steps) == 1:
        #     return possible_steps * 2
        # else:
        #     return possible_steps

    def check_deadlock(self):
        rail_env: RailEnv = self.unwrapped.rail_env
        # rail_env: RailEnv = self.unwrapped.rail_env
        new_deadlocked_agents = []

        for agent in rail_env.agents:

            if agent.status == RailAgentStatus.ACTIVE and agent.handle not in self._deadlocked_agents:
                position = agent.position
                direction = agent.direction

                while position is not None:

                    possible_transitions = rail_env.rail.get_transitions(*position, direction)
                    num_transitions = np.count_nonzero(possible_transitions)

                    if num_transitions == 1:  # if the agent only has one direction to go
                        new_direction_me = np.argmax(possible_transitions)
                        new_cell_me = get_new_position(position, new_direction_me)
                        opp_agent = rail_env.agent_positions[new_cell_me]

                        if opp_agent != -1:  # If nearby cell contains an opponent
                            opp_position = rail_env.agents[opp_agent].position
                            opp_direction = rail_env.agents[opp_agent].direction
                            opp_possible_transitions = rail_env.rail.get_transitions(*opp_position, opp_direction)
                            opp_num_transitions = np.count_nonzero(opp_possible_transitions)

                            if opp_num_transitions == 1:
                                if opp_direction != direction:  # If opposite direction as the current agent
                                    self._deadlocked_agents.append(agent.handle)
                                    new_deadlocked_agents.append(agent.handle)
                                    position = None
                                else:
                                    position = new_cell_me
                                    direction = new_direction_me
                            else:
                                position = new_cell_me
                                direction = new_direction_me
                        else:
                            position = None
                    else:
                        position = None

        return new_deadlocked_agents

    def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:

        rail_env: RailEnv = self.unwrapped.rail_env
        obs, reward, done, info = self.env.step(action_dict)
        o, d, i, r = {}, {}, {}, {}


        # award/penalize agents for their finishing status
        for agent_id, agent_obs in obs.items():
            o[agent_id] = obs[agent_id]
            d[agent_id] = done[agent_id]
            i[agent_id] = info[agent_id]

            if done[agent_id]:
                if rail_env.agents[agent_id].status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    # agent is done and really done -> give finished reward
                    r[agent_id] = self._finished_reward #- 0.05 * self.get_shortest_distance(rail_env, agent_id)
                else:
                    # agent is done but not really done -> give not_finished reward
                    r[agent_id] = self._not_finished_reward #- 0.05 * self.get_shortest_distance(rail_env, agent_id)
            else:
                r[agent_id] = 0

        d['__all__'] = done['__all__'] or all(d.values())

        return StepOutput(o, r, d, i)

    # def step(self, action_dict: Dict[int, RailEnvActions]) -> StepOutput:
    #
    #     rail_env: RailEnv = self.unwrapped.rail_env
    #     obs, reward, done, info = self.env.step(action_dict)
    #     o, d, i, r = {}, {}, {}, {}
    #
    #     # award/penalize agents for their finishing status
    #     for agent_id, agent_obs in obs.items():
    #         o[agent_id] = obs[agent_id]
    #         d[agent_id] = done[agent_id]
    #         i[agent_id] = info[agent_id]
    #
    #         if done[agent_id]:
    #             if rail_env.agents[agent_id].status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
    #                 # agent is done and really done -> give finished reward
    #                 r[agent_id] += self._finished_reward
    #             else:
    #                 # agent is done but not really done -> give not_finished reward
    #                 r[agent_id] += self._not_finished_reward
    #         else:
    #             r[agent_id] = -0.01 * self.get_shortest_distance(rail_env, agent_id)
    #
    #     if self._deadlock_reward != 0:
    #         new_deadlocked_agents = self.check_deadlock()
    #     else:
    #         new_deadlocked_agents = []
    #
    #     for agent_id, agent_obs in obs.items():
    #         # find deadlock agents, and kill them
    #         if agent_id not in self._deadlocked_agents or agent_id in new_deadlocked_agents:
    #
    #             if agent_id in new_deadlocked_agents:
    #                 # agent is in deadlocked (and was not before) -> give deadlock reward and set to done
    #                 r[agent_id] += self._deadlock_reward
    #                 d[agent_id] = True
    #
    #     d['__all__'] = done['__all__'] or all(d.values())
    #
    #     return StepOutput(o, r, d, i)

    def reset(self, random_seed: Optional[int] = None) -> Dict[int, Any]:
        self._deadlocked_agents = []
        return self.env.reset(random_seed)
