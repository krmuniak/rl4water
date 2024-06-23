from core.models.facility import Facility


class IrrigationDistrict(Facility):
    """
    Class to represent Irrigation District

    Attributes:
    ----------
    name : str
        identifier
    demand : float
    The list of monthly demand of the irrigation district
    total_deficit : float
    The total amount of water deficit we have
    list_deficits : list[float]
    The monthly list of the deficit of the irrigation district


    Methods:
    ----------
    determine_reward():
        Calculates the reward (irrigation deficit) given the values of its attributes
    determine_consumption():
        Determines how much water is consumed by the irrigation district
    determine_info():
        Returns info about the irrigation sustem
    """

    def __init__(self, name: str, all_demand: list[float], objective_function, objective_name: str) -> None:
        super().__init__(name, objective_function, objective_name)
        self.all_demand: list[float] = all_demand
        self.total_deficit: float = 0
        self.all_deficit: list[float] = []

    def get_current_demand(self) -> float:
        return self.all_demand[self.timestep % len(self.all_demand)]

    def determine_deficit(self) -> float:
        """
        Calculates the reward (irrigation deficit) given the values of its attributes

        Returns:
        ----------
        float
            Water deficit of the irrigation district
        """
        consumption = self.determine_consumption()
        deficit = self.get_current_demand() - consumption
        self.total_deficit += deficit
        self.all_deficit.append(deficit)
        return deficit

    def determine_reward(self) -> float:
        """
        Calculates the reward given the objective function for this district.
        Uses demand and received water.

        Returns:
        ----------
        float
            Reward for the objective function.
        """
        return self.objective_function(self.get_current_demand(), self.get_inflow(self.timestep))

    def determine_consumption(self) -> float:
        """
        Determines how much water is consumed by the irrigation district

        Returns:
        ----------
        float
            Water consumption
        """
        return min(self.get_current_demand(), self.get_inflow(self.timestep))

    def is_truncated(self) -> bool:
        return self.timestep >= len(self.all_demand)

    def determine_info(self) -> dict:
        """
        Determines info of irrigation district

        Returns:
        ----------
        dict
            Info about irrigation district (name, name, inflow, outflow, demand, timestep, deficit)
        """
        return {
            "name": self.name,
            "inflow": self.get_inflow(self.timestep),
            "outflow": self.get_outflow(self.timestep),
            "demand": self.get_current_demand(),
            "total_deficit": self.total_deficit,
            "list_deficits": self.all_deficit,
        }

    def reset(self) -> None:
        super().reset()
        self.total_deficit = 0
        self.all_deficit = []
