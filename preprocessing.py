import pandas as pd
import numpy as np
from typing import Tuple
from constants import cut_size


def take_a_cut(df: pd.DataFrame)->pd.DataFrame:
    """
    return a cut of the relevant data for testing
    """
    return df[df["index"] < cut_size()]


def read_in_input_data()->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    read in input data for...
    1. robot_locations
    2. range
    3. range_scenarios
    """

    robot_locations = pd.read_csv("robot_locations.csv")
    ranges = pd.read_csv("range.csv")
    ranges_scenarios = pd.read_csv("range_scenarios.csv")

    return (take_a_cut(robot_locations),take_a_cut(ranges),take_a_cut(ranges_scenarios))
