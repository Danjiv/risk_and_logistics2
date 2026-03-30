import xpress as xp
import numpy as np
import pandas as pd
from typing import Tuple


def MINLP_model(robot_locations_df: pd.DataFrame, ranges_df: pd.DataFrame, cut_size: int):
    """
    Solve the mixed-integer non-linear programming problem for a cut of the robots
    """

    robot_locations = robot_locations_df
    robot_range = ranges_df
    
    n_robots = cut_size
    n_stations = cut_size

    robots = range(n_robots)
    stations = range(n_stations)

    ###########################################################################################

    # Model constants

    max_chargers = 8

    max_bots_per_charger = 2

    annual_station_build_cost = 5000

    rescue_robot_cost = 1000

    maintenance_cost_per_charger = 500

    cost_per_km_of_charging = 0.42

    range_min = 10

    range_max = 175

    ###########################################################################################

    # Build optimization model

    ###########################################################################################

    prob = xp.problem("MINLP")

    xp.setOutputEnabled(False)
    # prob.controls.maxtime = -300

    ##########################################################################################

    # Declarations

    ##########################################################################################

    # binary variable - y_ij = 1 if robot i is within range of charging station j

    y = np.array([prob.addVariable(name = 'y_{0}_{1}'.format(r, s), vartype = xp.binary)
                  for r in robots for s in stations], dtype = xp.npvar).reshape(n_robots, n_stations)
    
    # binary variable - x_ij = 1 if robot i is assigned to charge at station j

    x = np.array([prob.addVariable(name = 'x_{0}_{1}'.format(r, s), vartype = xp.binary)
                  for r in robots for s in stations], dtype = xp.npvar).reshape(n_robots, n_stations)

    # binary variable - o_j = 1 if we open station j

    o = np.array([prob.addVariable(name = 'o_{0}'.format(s), vartype = xp.binary)
                  for s in stations], dtype = xp.npvar).reshape(n_stations)
    
    # integer variable - p_j = the number of charging points to open at station j

    p = np.array([prob.addVariable(name = 'p_{0}'.format(s), vartype = xp.integer, ub = 8)
                  for s in stations], dtype = xp.npvar).reshape(n_stations)

    # continous variable - x_coord_j is the x_coordinate of station j
    x_coord = np.array([prob.addVariable(name = 'x_coord_{0}'.format(s), vartype = xp.continuous)
                       for s in stations], dtype = xp.npvar).reshape(n_stations)

    # continuous variable - y_coord_j is the y_coordinate of station j    
    y_coord = np.array([prob.addVariable(name = 'y_coord_{0}'.format(s), vartype = xp.continuous)
                       for s in stations], dtype = xp.npvar).reshape(n_stations)
    
    # continuous variable - d_ij is the distance of robot i from station j

    d = np.array([prob.addVariable(name = 'd_{0}_{1}'.format(r, s), vartype = xp.continuous)
                  for r in robots for s in stations], dtype = xp.npvar).reshape(n_robots, n_stations)
    
    # binary variable - c_ij = 1 if we have assigned robot i to station j and it does not have the range to make it

    c = np.array([prob.addVariable(name = 'c_{0}_{1}'.format(r, s), vartype = xp.binary)
                  for r in robots for s in stations], dtype = xp.npvar).reshape(n_robots, n_stations)
    
    #################################################################################################################

    # Objective function

    #################################################################################################################

    prob.setObjective(xp.Sum(x[i,j]*cost_per_km_of_charging*(range_max - (robot_range["range"][i] - d[i,j])) for i in robots for j in stations) +
                      xp.Sum(o[j]*annual_station_build_cost for j in stations) + 
                      xp.Sum(p[j]*maintenance_cost_per_charger for j in stations) +
                      xp.Sum(c[i, j]*rescue_robot_cost for i in robots for j in stations) +
                      xp.Sum(c[i,j]*cost_per_km_of_charging*(range_max - robot_range["range"][i]) for i in robots for j in stations),                      
                      sense = xp.minimize)
    
    ##################################################################################################################

    # Constraints

    ##################################################################################################################

    # each robot needs to be assigned to a charging station

    prob.addConstraint(xp.Sum(x[i, j] for j in stations)==1 for i in robots)

    # a charging station must be open for any robots to be assigned to charge at it

    prob.addConstraint(x[i, j] <= o[j] for i in robots for j in stations)

    # number of robots assigned to a charging station cannot exceed the number of available chargers

    prob.addConstraint(xp.Sum(x[i, j] for i in robots) <= p[j]*max_bots_per_charger*o[j] for j in stations)

    # set the distance of each robot from each charging station

    prob.addConstraint(d[i, j] == xp.sqrt((y_coord[j] - robot_locations["latitude"][i])**2 + 
                                         (x_coord[j] - robot_locations["longitude"][i])**2)
                                         for i in robots for j in stations)
    
    # set the y variable - can robot i reach station j

    prob.addConstraint(robot_range["range"][i] - d[i, j] <= range_max*y[i, j] for i in robots for j in stations)


    # set c, the robot needs rescuing variable

    prob.addConstraint(x[i, j] - y[i, j] <= c[i, j] for i in robots for j in stations)


    prob.solve()

    print("Solution status:", prob.attributes.solstatus)
    #Values
    # 0 XPRS_SOLSTATUS_NOTFOUND No solution available.
    # 1 XPRS_SOLSTATUS_OPTIMAL An optimal solution has been found.
    # 2 XPRS_SOLSTATUS_FEASIBLE A solution that is not proven optimal is found.
    # 3 XPRS_SOLSTATUS_INFEASIBLE No solution exists.
    # 4 XPRS_SOLSTATUS_UNBOUNDED The problem is unbounded, if feasible

    objective_function_val = prob.attributes.objval
    print(f"Objective function value: {objective_function_val}")

    print(prob.getSolution(x))
    print(prob.getSolution(o))
    print(prob.getSolution(y))
    print(prob.getSolution(c))
    print(prob.getSolution(p))