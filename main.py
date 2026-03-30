import pandas as pd
import numpy as np
from preprocessing import read_in_input_data
from MINLP_model import MINLP_model
from constants import cut_size


def main():
    robot_locations, ranges, ranges_scenarios = read_in_input_data()


    MINLP_model(robot_locations, ranges, cut_size())



if __name__ == "__main__":
    main()