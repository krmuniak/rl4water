import gymnasium as gym
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from gymnasium.spaces import Box, Dict, Space
from gymnasium.core import ObsType, RenderFrame
from typing import Any, Union, Optional
from core.models.flow import Flow
from core.models.facility import Facility, ControlledFacility


class WaterManagementSystem(gym.Env):
    def __init__(
        self,
        water_systems: list[Union[Facility, ControlledFacility, Flow]],
        rewards: dict,
        start_date: datetime,
        timestep_size: relativedelta,
        seed: int = 42,
    ) -> None:
        self.water_systems: list[Union[Facility, ControlledFacility, Flow]] = water_systems
        self.rewards: dict = rewards

        self.start_date: datetime = start_date
        self.current_date: datetime = start_date
        self.timestep_size: relativedelta = timestep_size
        self.timestep: int = 0

        self.seed: int = seed

        self.observation_space: Space = self._determine_observation_space()
        self.action_space: Space = self._determine_action_space()
        self.reward_space: Space = Box(-np.inf, np.inf, shape=(len(rewards.keys()),))

        self.observation: np.array = self._determine_observation()

        for water_system in self.water_systems:
            water_system.current_date = self.current_date
            water_system.timestep_size = self.timestep_size

    def _determine_observation(self) -> np.array:
        result = []
        for water_system in self.water_systems:
            if isinstance(water_system, ControlledFacility):
                result.append(water_system.determine_observation())
        return np.array(result)

    def _determine_observation_space(self) -> Dict:
        return Dict(
            {
                water_system.name: water_system.observation_space
                for water_system in self.water_systems
                if isinstance(water_system, ControlledFacility)
            }
        )

    def _determine_action_space(self) -> Dict:
        return Dict(
            {
                water_system.name: water_system.action_space
                for water_system in self.water_systems
                if isinstance(water_system, ControlledFacility)
            }
        )

    def _is_truncated(self) -> bool:
        return False

    def _determine_info(self) -> dict[str, Any]:
        # TODO: decide on what we wnat to output in the info.
        return {"water_systems": self.water_systems}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[ObsType, dict[str, Any]]:
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)
        self.current_date = self.start_date
        self.timestep = 0

        self.observation: np.array = self._determine_observation()
        # Reset rewards
        for key in self.rewards.keys():
            self.rewards[key] = 0

        for water_system in self.water_systems:
            water_system.current_date = self.start_date
            water_system.reset()
        return self.observation, self._determine_info()

    def step(self, action: np.array) -> tuple[np.array, np.array, bool, bool, dict]:
        final_reward = {}

        # Reset rewards
        for key in self.rewards.keys():
            final_reward[key] = 0

        final_observation = {}
        final_terminated = False
        final_truncated = False
        final_info = {"date": self.current_date}

        for water_system in self.water_systems:
            water_system.current_date = self.current_date

            if isinstance(water_system, ControlledFacility):
                observation, reward, terminated, truncated, info = water_system.step(action[water_system.name])
            elif isinstance(water_system, Facility) or isinstance(water_system, Flow):
                observation, reward, terminated, truncated, info = water_system.step()
            else:
                raise ValueError()

            # Set observation for a Controlled Facility.
            if isinstance(water_system, ControlledFacility):
                final_observation[water_system.name] = observation

            # Add reward to the objective assigned to this Facility (unless it is a Flow).
            if isinstance(water_system, Facility) or isinstance(water_system, ControlledFacility):
                if water_system.objective_name:
                    final_reward[water_system.objective_name] += reward

            # Store additional information
            final_info[water_system.name] = info

            # Determine whether program should stop
            final_terminated = final_terminated or terminated
            final_truncated = final_truncated or truncated or self._is_truncated()

            if final_terminated or final_truncated:
                break

        self.timestep += 1
        self.current_date += self.timestep_size

        return (
            np.array(list(final_observation.values())).flatten(),
            np.array(list(final_reward.values())).flatten(),
            final_terminated,
            final_truncated,
            final_info,
        )

    def close(self) -> None:
        # TODO: implement if needed, e.g. for closing opened rendering frames.
        pass

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        # TODO: implement if needed, for rendering simulation.
        pass
