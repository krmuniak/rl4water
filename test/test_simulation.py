import calendar
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from pytest import approx
from pathlib import Path
from examples.nile_river_simulation import create_nile_river_env
from core.utils.utils import convert_str_to_float_list

data_directory = Path(__file__).parent

FILE_MASTER_SIMULATION_RUN = data_directory / "data" / "master.csv"

COLUMN_ACTION_TO_RUN = "Input"
COLUMNS_INFO_TO_VERIFY = [
    "Gerd_storage",
    "Gerd_release",
    "Roseires_storage",
    "Roseires_release",
    "Sennar_storage",
    "Sennar_release",
    "Had_storage",
    "Had_release",
    "Gerd_production",
]


def test_simulation() -> None:
    water_management_system = create_nile_river_env()
    water_management_system.reset()

    simulation_run = load_simulation_run_from_file(
        FILE_MASTER_SIMULATION_RUN, COLUMN_ACTION_TO_RUN, COLUMNS_INFO_TO_VERIFY
    )

    for simulation_step in simulation_run:
        action = np.array(convert_str_to_float_list(simulation_step["action"]))
        info_to_verify = simulation_step["info"]

        _, _, _, _, final_info = water_management_system.step(action)

        # Skip leap years as they were not supported in the original Nile River simulation
        if calendar.isleap(water_management_system.unwrapped.current_date.year):
            water_management_system.unwrapped.current_date += relativedelta(years=1)

        assert get_info_for_verification(final_info) == approx(info_to_verify, abs=0.1)


def load_simulation_run_from_file(
    file_name: Path, column_action_to_run: str, columns_to_verify: list[str]
) -> list[dict]:
    simulation_run_file = pd.read_csv(file_name)

    simulation_run = []

    # Iterate over each row and column, comparing values to 3 decimal points
    for _, row in simulation_run_file.iterrows():
        simulation_step = {
            "action": row[column_action_to_run],
            "info": [row[column_to_verify] for column_to_verify in columns_to_verify],
        }

        simulation_run.append(simulation_step)

    return simulation_run


def get_info_for_verification(final_info: dict) -> list[float]:
    return [
        final_info["GERD"]["stored_water"],
        final_info["GERD"]["current_release"],
        final_info["Roseires"]["stored_water"],
        final_info["Roseires"]["current_release"],
        final_info["Sennar"]["stored_water"],
        final_info["Sennar"]["current_release"],
        final_info["HAD"]["stored_water"],
        final_info["HAD"]["current_release"],
        final_info["GERD_power_plant"]["monthly_production"],
    ]
