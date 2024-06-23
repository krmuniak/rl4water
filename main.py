import pprint
import numpy as np
from examples.nile_river_simulation import create_nile_river_env
import csv

make_csv = True


def nile_river_simulation(nu_of_timesteps=240):
    # Create power plant, reservoir and irrigation district. Initialise with semi-random parameters.
    # Set objective functions to identity for power plant, minimum_water_level for reservoir and water_deficit_minimised
    # for irrigation district.

    water_management_system = create_nile_river_env()

    if make_csv:
        with open("verification/group13.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Year",
                    "Input",
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
            )
            np.random.seed(42)
            for i in range(nu_of_timesteps):
                action = generateOutput()
                (
                    final_observation,
                    final_reward,
                    final_terminated,
                    final_truncated,
                    final_info,
                ) = water_management_system.step(action)
                writer.writerow(
                    [
                        i,
                        action,
                        ensure_float(final_info.get("GERD")["stored_water"]),
                        ensure_float(final_info.get("GERD")["current_release"]),
                        ensure_float(final_info.get("Roseires")["stored_water"]),
                        ensure_float(final_info.get("Roseires")["current_release"]),
                        ensure_float(final_info.get("Sennar")["stored_water"]),
                        ensure_float(final_info.get("Sennar")["current_release"]),
                        ensure_float(final_info.get("HAD")["stored_water"]),
                        ensure_float(final_info.get("HAD")["current_release"]),
                        ensure_float(final_info.get("GERD_power_plant")["monthly_production"]),
                    ]
                )
    else:
        for _ in range(nu_of_timesteps):
            action = water_management_system.action_space.sample()
            action = np.array(list(action.values())).flatten()
            print("Action:", action, "\n")
            (
                final_observation,
                final_reward,
                final_terminated,
                final_truncated,
                final_info,
            ) = water_management_system.step(action)
            print("Reward:", final_reward)
            pprint.pprint(final_info)
            print("Is finished:", final_truncated, final_terminated)


def generateOutput():
    random_values = np.random.rand(
        4,
    ) * [10000, 10000, 10000, 4000]

    return random_values


def ensure_float(value):
    if isinstance(value, np.ndarray):
        return value.item()
    return value


if __name__ == "__main__":
    nile_river_simulation()
