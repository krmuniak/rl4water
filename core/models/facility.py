import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from dateutil.relativedelta import relativedelta
from gymnasium.spaces import Space
from gymnasium.core import ObsType, ActType
from typing import SupportsFloat, Optional
from core.models.objective import Objective


class Facility(ABC):
    def __init__(self, name: str, objective_function=Objective.no_objective, objective_name: str = "") -> None:
        self.name: str = name
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.current_date: Optional[datetime] = None
        self.timestep_size: Optional[relativedelta] = None
        self.timestep: int = 0

        self.split_release = None

    @abstractmethod
    def determine_reward(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_consumption(self) -> float:
        raise NotImplementedError()

    def is_terminated(self) -> bool:
        return False

    def is_truncated(self) -> bool:
        return False

    def get_inflow(self, timestep: int) -> float:
        return self.all_inflow[timestep]

    def set_inflow(self, timestep: int, inflow: float) -> None:
        if len(self.all_inflow) == timestep:
            self.all_inflow.append(inflow)
        elif len(self.all_inflow) > timestep:
            self.all_inflow[timestep] += inflow
        else:
            raise IndexError

    def determine_outflow(self) -> float:
        return self.get_inflow(self.timestep) - self.determine_consumption()

    def get_outflow(self, timestep: int) -> float:
        return self.all_outflow[timestep]

    def step(self) -> tuple[ObsType, float, bool, bool, dict]:
        self.all_outflow.append(self.determine_outflow())
        # TODO: Determine if we need to satisy any terminating codnitions for facility.
        reward = self.determine_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.determine_info()

        self.timestep += 1

        return None, reward, terminated, truncated, info

    def reset(self) -> None:
        self.timestep: int = 0
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []

    def determine_info(self) -> dict:
        raise {}

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class ControlledFacility(ABC):
    def __init__(
        self,
        name: str,
        observation_space: Space,
        action_space: ActType,
        objective_function=Objective.no_objective,
        objective_name: str = "",
        max_capacity: float = float("inf"),
    ) -> None:
        self.name: str = name
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []

        self.observation_space: Space = observation_space
        self.action_space: Space = action_space

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.max_capacity: float = max_capacity

        self.current_date: Optional[datetime] = None
        self.timestep_size: Optional[relativedelta] = None
        self.timestep: int = 0

        self.should_split_release = np.prod(self.action_space.shape) > 1
        self.split_release = None

    @abstractmethod
    def determine_reward(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_outflow(action: ActType) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_observation(self) -> ObsType:
        raise NotImplementedError()

    @abstractmethod
    def is_terminated(self) -> bool:
        raise NotImplementedError()

    def is_truncated(self) -> bool:
        return False

    def get_inflow(self, timestep: int) -> float:
        return self.all_inflow[timestep]

    def set_inflow(self, timestep: int, inflow: float) -> None:
        if len(self.all_inflow) == timestep:
            self.all_inflow.append(inflow)
        elif len(self.all_inflow) > timestep:
            self.all_inflow[timestep] += inflow
        else:
            raise IndexError

    def get_outflow(self, timestep: int) -> float:
        return self.all_outflow[timestep]

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        self.all_outflow.append(self.determine_outflow(action))
        # TODO: Change stored_water to multiple outflows.

        observation = self.determine_observation()
        reward = self.determine_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.determine_info()

        self.timestep += 1

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self) -> None:
        self.timestep: int = 0
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []

    def determine_info(self) -> dict:
        raise {}

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)
