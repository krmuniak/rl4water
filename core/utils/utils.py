import numpy as np

LIST_SPECIFIC_CHARACTERS = "[],"


def generate_random_actions(number_of_actions=4, seed=42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.rand(4) * [10000, 10000, 10000, 4000]


def convert_str_to_float_list(string_list: str) -> list:
    return list(map(float, string_list.translate(str.maketrans("", "", LIST_SPECIFIC_CHARACTERS)).split()))
