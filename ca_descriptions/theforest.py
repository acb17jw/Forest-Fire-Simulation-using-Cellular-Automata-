import inspect
import math
import random
import sys
import numpy as np

this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')

import capyle.utils as utils
from capyle.ca import Grid2D, Neighbourhood, randomise2d


# constants
WORLD_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
GRID_SIZE = 100
NUM_GENERATION = 500
WIND_DIRECTION = "SW"
WIND_SPEED = 10

# water drop variables
global water_counter
water_counter = 180
water_drop_up = 0
water_drop_down = 20
water_drop_left = 80
water_drop_right = 100
water_drop_time = 26

# cell states
CHAPARRAL = 0
FOREST = 1
LAKE = 2
CANYON = 3
BURNING = 4
BURNT_ALREADY = 5
BURNING_START = 6
BURNING_ENDING = 7

# burning tresholds
start_burning_threshhold = [0.02, 0.005, 0, 0.05, 0, 0, 0.04]
start_burning_factor = 20
burning_threshhold = [0.04, 0.01, 0, 0.1, 0, 0, 0.04]
burning_factor = 30

# extinguishing values
global ext_val
ext_val = [0, 0, 0, 0, 0]
ext_val[CHAPARRAL] = 62
ext_val[FOREST] = 150
ext_val[LAKE] = 1
ext_val[CANYON] = 2
# for initial state, only burning cell is chaparral
ext_val[BURNING] = ext_val[CHAPARRAL]


# start grid and ignition_grid
global start_grid
global ignition_grid

# seting up start grid
start_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
start_grid[60:80, 30:50] = FOREST
start_grid[20:30, 10:30] = LAKE
start_grid[10:60, 60:70] = CANYON
start_grid[0, GRID_SIZE-1] = BURNING  # initial fire right upper corner
ignition_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)


def setup(args):
    config_path = args[0]
    config = utils.load(config_path)
    config.title = "The forest"
    config.dimensions = 2
    config.grid_dims = (GRID_SIZE, GRID_SIZE)
    config.num_generations = NUM_GENERATION
    config.states = (CHAPARRAL, FOREST, LAKE, CANYON, BURNING,
                     BURNT_ALREADY, BURNING_START, BURNING_ENDING)
    config.state_colors = \
        [
            (0.7, 0.7, 0.1),  # chaparral
            (0, 0.6, 0),  # forrest
            (0, 0.5, 1),  # lake
            (1, 0.6, 0.1),  # canyon
            (1, 0, 0),  # burning
            (0.25, 0.25, 0.25),  # burnt already
            (1, 0.7, 0),  # burn start
            (0.8, 0, 0.2)  # burning end
        ]
    config.set_initial_grid(start_grid)
    config.wrap = False

    if len(args) == 2:
        config.save()
        sys.exit()
    return config


def transition_function(grid, neighbourstates, neighbourcounts, ext_grid):
    global water_counter
    global ignition_grid

    neighbourstates = np.array(neighbourstates)
    init_grid = start_grid.astype(int)
    iggrid = np.array(ignition_grid)

    # handle fire ignition factors
    igfactors = []
    for i in range(len(grid)):
        row = []
        for j in range(len(grid[i])):
            row.append(first_phase(grid[i][j], neighbourstates[:, i, j], calculate_wind()))
        igfactors.append(row)
    igfactors = np.array(igfactors)


    # handle start burn state
    already_started_to_burn = []
    for i in range(len(grid)):
        row = []
        for j in range(len(grid[i])):
            row.append(second_phase(grid[i][j], iggrid[i, j], igfactors[i, j]))
        already_started_to_burn.append(row)
    grid[already_started_to_burn] = BURNING_START

    #handle buning statr
    iggrid = np.add(igfactors, iggrid)   
    burning = []
    for i in range(len(grid)):
        row = []
        for j in range(len(grid[i])):
            row.append(third_phase( grid[i][j], iggrid[i, j], ext_grid[i, j]))
        burning.append(row)
    grid[burning] = BURNING


    # handle end burning state
    end_burn = []
    for i in range(len(grid)):
        row = []
        for j in range(len(grid[i])):
            row.append(fourth_phase( grid[i][j], ext_grid[i, j], int(start_grid[i, j])) )
        end_burn.append(row)

    grid[end_burn] = BURNING_ENDING

    ext_grid[(grid == BURNING) | (grid == BURNING_ENDING)] -= 1
    already_burnt = (ext_grid == 0)
    grid[already_burnt] = BURNT_ALREADY

    water_counter += 1

    if(water_counter> water_drop_time and water_counter < water_drop_time+100):
        grid[water_drop_up:water_drop_down, water_drop_left:water_drop_right] = LAKE

    if(water_counter == water_drop_time+100):
        grid[water_drop_up:water_drop_down, water_drop_left:water_drop_right] = start_grid[water_drop_up:water_drop_down, water_drop_left:water_drop_right]

    ignition_grid = iggrid
    return grid


def first_phase(state, neighbourstates, wind_factor):
    current_state = int(state)
    neighbourstates = neighbourstates.astype(int)
    if(current_state == BURNING or current_state == LAKE or current_state == BURNT_ALREADY or current_state == BURNING_ENDING):
        return 0
    fire_factor = 0
    for i, nbstate in enumerate(neighbourstates):

        ran = random.uniform(0, 1)

        if(nbstate == BURNING and burning_threshhold[current_state] * wind_factor[i] >= ran):
            f = math.floor(burning_factor * wind_factor[i])
            fire_factor += int(f)

        if(nbstate == BURNT_ALREADY or nbstate == BURNING_ENDING):
            if(start_burning_threshhold[current_state] * wind_factor[i] >= ran):
                f = math.floor(start_burning_factor * wind_factor[i])
                fire_factor += int(f)

    if(current_state == BURNING_START):
        fire_factor += start_burning_factor

    return int(fire_factor)

def second_phase(state, iggrid_state, start_burning_grid_state):
    if(state == BURNING_START):
        return True
    if(state != BURNING and state != LAKE and state != BURNING_ENDING and state != BURNT_ALREADY and iggrid_state == 0 and start_burning_grid_state > 0):
        return True
    return False


def third_phase(state, iggrid_state, ext_grid_state):
    if(state == BURNING):
        return True
    if(state == BURNING_START and iggrid_state >= ext_grid_state):
        return True
    return False

def fourth_phase(state, ext_grid_state, initial_grid_state):
    if(state == BURNING_ENDING):
        return True
    if(state == BURNING and ext_val[initial_grid_state] >= ext_grid_state * 2):
        return True
    return False


def calculate_wind():
    wind_factor = np.zeros(8)
    angl = 0
    for i in range(8):
        wind_factor[(i + WORLD_DIRECTIONS.index(WIND_DIRECTION) ) % 8] = np.exp(
            WIND_SPEED * np.cos(np.deg2rad(angl)) * 0.1783)
        angl += 45

    indexation = [
            3, 4, 5, 2, 6, 1, 0, 7
        ]
    return wind_factor[indexation]


def main():

    config = setup(sys.argv[1:])
    ext_grid = [[ext_val[i] for i in j]
                  for j in start_grid.astype(int)]
    ext_grid = np.array(ext_grid)

    ignition_grid = np.zeros((GRID_SIZE, GRID_SIZE))
    ignition_grid = ignition_grid.astype(int)
    grid = Grid2D(config, (transition_function, ext_grid))
    timeline = grid.run()

    config.save()
    utils.save(timeline, config.timeline_path) 


if __name__ == "__main__":
    main()
