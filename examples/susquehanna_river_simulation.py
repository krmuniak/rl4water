import numpy as np
from pathlib import Path
from gymnasium.spaces import Box
from gymnasium.wrappers.time_limit import TimeLimit
from datetime import datetime
from dateutil.relativedelta import relativedelta
from core.envs.water_management_system import WaterManagementSystem
from core.models.reservoir import Reservoir
from core.models.flow import Flow, Inflow
from core.models.objective import Objective
from core.models.power_plant import PowerPlant
from core.models.irrigation_district import IrrigationDistrict
from core.wrappers.transform_action import ReshapeArrayAction

data_directory = Path(__file__).parents[1] / "examples" / "data" / "susquehanna_river"


def create_susquehanna_river_env() -> WaterManagementSystem:
    class ReservoirDateDependendObjetive(Reservoir):
        def determine_reward(self) -> float:
            is_weekend = self.current_date.weekday() < 5

            return self.objective_function(is_weekend, self.storage_to_level(self.stored_water))

    class PowerPlantSequentialObjetive(PowerPlant):
        def determine_reward(self) -> float:
            return self.objective_function(self.timestep, self.determine_production())

    Conowingo_reservoir = ReservoirDateDependendObjetive(
        name="Conowingo",
        observation_space=Box(low=0, high=3279501720),
        action_space=Box(low=0, high=1242857, shape=(4,)),
        integration_timestep_size=relativedelta(hours=1),
        objective_function=Objective.is_greater_than_minimum_with_condition(106.5),
        objective_name="recreation",
        stored_water=2641905256.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_Conowingo.txt"),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_Conowingo.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_Conowingo.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_sur_rel_Conowingo.txt"),
    )

    Power_plant = PowerPlantSequentialObjetive(
        name="Power_plant",
        objective_function=Objective.sequential_scalar(np.loadtxt(data_directory / "reservoirs" / "energy_prices.txt")),
        objective_name="energy_revenue",
        efficiency=0.79,
        min_turbine_flow=210.0,
        max_turbine_flow=85412.0,
        head_start_level=0,
        max_capacity=float("inf"),
        reservoir=Conowingo_reservoir,
    )

    Atomic_system = IrrigationDistrict(
        name="Atomic",
        all_demand=np.loadtxt(data_directory / "demands" / "Atomic.txt"),
        objective_function=Objective.supply_ratio_maximised,
        objective_name="water_supply",
    )

    Baltimore_system = IrrigationDistrict(
        name="Baltimore",
        all_demand=np.loadtxt(data_directory / "demands" / "Baltimore.txt"),
        objective_function=Objective.supply_ratio_maximised,
        objective_name="water_supply",
    )

    Chester_system = IrrigationDistrict(
        name="Chester",
        all_demand=np.loadtxt(data_directory / "demands" / "Chester.txt"),
        objective_function=Objective.supply_ratio_maximised,
        objective_name="water_supply",
    )

    Downstream_system = IrrigationDistrict(
        name="Downstream",
        all_demand=np.loadtxt(data_directory / "demands" / "Downstream.txt"),
        objective_function=Objective.deficit_squared_ratio_minimised,
        objective_name="enviromental_shortage",
    )

    Conowingo_inflow_main = Inflow(
        name="conowingo_inflow_main",
        destinations=Conowingo_reservoir,
        max_capacity=float("inf"),
        all_inflow=np.loadtxt(data_directory / "inflows" / "InflowConowingoMain.txt"),
    )

    Conowingo_inflow_lateral = Inflow(
        name="conowingo_inflow_lateral",
        destinations=Conowingo_reservoir,
        max_capacity=float("inf"),
        all_inflow=np.loadtxt(data_directory / "inflows" / "InflowConowingoLateral.txt"),
    )

    Conowingo_outflow = Flow(
        name="conowingo_outflow",
        sources=[Conowingo_reservoir],
        destinations={Atomic_system: 0.25, Baltimore_system: 0.25, Chester_system: 0.25, Power_plant: 0.25}, 
        max_capacity=float("inf"),
    )

    Donwstream_inflow = Flow(
        name="donwstream_inflow",
        sources=[Power_plant],
        destinations=Downstream_system,
        max_capacity=float("inf"),
    )

    # Muddy_reservoir = Reservoir(
    #     name="Muddy",
    #     observation_space=Box(low=0, high=2471202360),
    #     action_space=Box(low=0, high=32000),
    #     integration_timestep_size=relativedelta(hours=1),
    #     objective_function=Objective.no_objective,
    #     objective_name="",
    #     stored_water=0,
    #     evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_Muddy.txt"),
    #     storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_Muddy.txt"),
    #     storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_Muddy.txt"),
    #     storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_sur_rel_Muddy.txt"),
    # )

    water_management_system = WaterManagementSystem(
        water_systems=[
            Conowingo_inflow_main,
            Conowingo_inflow_lateral,
            Conowingo_reservoir,
            Conowingo_outflow,
            Atomic_system,
            Baltimore_system,
            Chester_system,
            Power_plant,
            Donwstream_inflow,
            Downstream_system,
        ],
        rewards={
            "recreation": 0,
            "energy_revenue": 0,
            "water_supply": 0,
            "enviromental_shortage": 0,
        },
        start_date=datetime(2025, 1, 1),
        timestep_size=relativedelta(hours=4),
        seed=42,
    )

    water_management_system = ReshapeArrayAction(water_management_system)
    water_management_system = TimeLimit(water_management_system, max_episode_steps=2190)

    return water_management_system
