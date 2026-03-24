import pandas as pd
import numpy as np
from preprocessing import read_in_input_data


def main():
    robot_locations, ranges, ranges_scenarios = read_in_input_data()

    print(robot_locations.shape[0])
    print(ranges.shape[0])
    print(ranges_scenarios.shape[0])


if __name__ == "__main__":
    main()