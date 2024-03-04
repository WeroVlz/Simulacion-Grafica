"""
    Team names:
        Edgar Velazquez Mercado
        Pablo Raschid Llamas Aun
        José Iván Andrade Rojas
"""

from argparse import ArgumentParser, Namespace
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, datetime

first_iteration = True


def define_patterns():
    patterns = {
        "Block": [np.array([[0, 0, 0, 0],
                            [0, 255, 255, 0],
                            [0, 255, 255, 0],
                            [0, 0, 0, 0]])],
        "Beehive": [np.array([[0, 0, 0, 0, 0, 0],
                              [0, 0, 255, 255, 0, 0],
                              [0, 255, 0, 0, 255, 0],
                              [0, 0, 255, 255, 0, 0],
                              [0, 0, 0, 0, 0, 0]])],
        "Loaf": [np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 255, 255, 0, 0],
                           [0, 255, 0, 0, 255, 0],
                           [0, 0, 255, 0, 255, 0],
                           [0, 0, 0, 255, 0, 0],
                           [0, 0, 0, 0, 0, 0]])],
        "Boat": [np.array([[0, 0, 0, 0, 0],
                           [0, 255, 255, 0, 0],
                           [0, 255, 0, 255, 0],
                           [0, 0, 255, 0, 0],
                           [0, 0, 0, 0, 0]])],
        "Tub": [np.array([[0, 0, 0, 0, 0],
                          [0, 0, 255, 0, 0],
                          [0, 255, 0, 255, 0],
                          [0, 0, 255, 0, 0],
                          [0, 0, 0, 0, 0]])],
        "Blinker": [np.array([[0, 0, 0, 0, 0],
                              [0, 0, 255, 0, 0],
                              [0, 0, 255, 0, 0],
                              [0, 0, 255, 0, 0],
                              [0, 0, 0, 0, 0]]),
                    np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 255, 255, 255, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]])],
        "Toad": [np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 255, 0, 0],
                           [0, 255, 0, 0, 255, 0],
                           [0, 255, 0, 0, 255, 0],
                           [0, 0, 255, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]]),
                 np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 255, 255, 255, 0],
                           [0, 255, 255, 255, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])],
        "Beacon": [np.array([[0, 0, 0, 0, 0, 0],
                             [0, 255, 255, 0, 0, 0],
                             [0, 255, 255, 0, 0, 0],
                             [0, 0, 0, 255, 255, 0],
                             [0, 0, 0, 255, 255, 0],
                             [0, 0, 0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0, 0, 0],
                             [0, 255, 255, 0, 0, 0],
                             [0, 255, 0, 0, 0, 0],
                             [0, 0, 0, 0, 255, 0],
                             [0, 0, 0, 255, 255, 0],
                             [0, 0, 0, 0, 0, 0]])],
        "Glider": [np.array([[0, 0, 0, 0, 0],
                             [0, 0, 255, 0, 0],
                             [0, 0, 0, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0, 0],
                             [0, 255, 0, 255, 0],
                             [0, 0, 255, 255, 0],
                             [0, 0, 255, 0, 0],
                             [0, 0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 255, 0],
                             [0, 255, 0, 255, 0],
                             [0, 0, 255, 255, 0],
                             [0, 0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0, 0],
                             [0, 255, 0, 0, 0],
                             [0, 0, 255, 255, 0],
                             [0, 255, 255, 0, 0],
                             [0, 0, 0, 0, 0]])
                   ],
        "LW Spaceship": [np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 255, 0, 0, 255, 0, 0],
                                   [0, 0, 0, 0, 0, 255, 0],
                                   [0, 255, 0, 0, 0, 255, 0],
                                   [0, 0, 255, 255, 255, 255, 0],
                                   [0, 0, 0, 0, 0, 0, 0]]),
                         np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 255, 255, 0, 0],
                                   [0, 255, 255, 0, 255, 255, 0],
                                   [0, 255, 255, 255, 255, 0, 0],
                                   [0, 0, 255, 255, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]]),
                         np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 255, 255, 255, 255, 0],
                                   [0, 255, 0, 0, 0, 255, 0],
                                   [0, 0, 0, 0, 0, 255, 0],
                                   [0, 255, 0, 0, 255, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]]),
                         np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 255, 255, 0, 0, 0],
                                   [0, 255, 255, 255, 255, 0, 0],
                                   [0, 255, 255, 0, 255, 255, 0],
                                   [0, 0, 0, 255, 255, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]])
                         ]
    }
    return patterns


def detect_patterns(grid, width, height):
    patterns = define_patterns()
    rotate_3_times = ["Loaf", "Boat", "Glider", "LW Spaceship"]
    rotate_1_time = ["Beehive", "Toad", "Beacon"]
    pattern_count = {}
    for pattern in patterns:
        count = 0
        for p in patterns.get(pattern):
            if pattern in rotate_3_times:
                for i in range(4):
                    p_width, p_height = p.shape
                    for j in range(width - p_width + 1):
                        for k in range(height - p_height + 1):
                            if np.all(grid[j:j + p_width, k:k + p_height] == p):
                                count += 1
                    p = np.rot90(p, k=1)
            elif pattern in rotate_1_time:
                for i in range(2):
                    p_width, p_height = p.shape
                    for i in range(width - p_width + 1):
                        for j in range(height - p_height + 1):
                            if np.all(grid[i:i + p_width, j:j + p_height] == p):
                                count += 1
                    p = np.rot90(p, k=1)
            else:
                p_width, p_height = p.shape
                for i in range(width - p_width + 1):
                    for j in range(height - p_height + 1):
                        if np.all(grid[i:i + p_width, j:j + p_height] == p):
                            count += 1
        pattern_count[pattern] = count

    return pattern_count


def generate_grid(width, height, cells_array):
    grid = np.zeros((width, height), dtype=np.uint8)
    for cell in cells_array:
        grid[cell[0], cell[1]] = 255
    return grid


def check_neighbors(updated_grid, grid, width, height):
    around = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]]
    for row_index, row in enumerate(grid):
        for cell_index, cell in enumerate(row):
            live_neighbours = 0
            for neighbor in around:
                if 0 <= row_index + neighbor[0] <= width - 1 and 0 <= cell_index + neighbor[1] <= height - 1:
                    if grid[row_index + neighbor[0]][cell_index + neighbor[1]] == 255:
                        live_neighbours += 1
            if grid[row_index][cell_index] == 255:
                if not (live_neighbours == 2 or live_neighbours == 3):
                    updated_grid[row_index][cell_index] = 0

            if grid[row_index][cell_index] == 0:
                if live_neighbours == 3:
                    updated_grid[row_index][cell_index] = 255
    return updated_grid


def generate_iteration_log(frame, output_file, pattern_count):
    max_pattern_length = max(len(pattern) for pattern in pattern_count)
    total_pattern_count = 0

    for count in pattern_count:
        total_pattern_count += pattern_count.get(count)

    with open(output_file, "a") as file:
        file.write(f'\n\nIteration: {frame + 1}')
        file.write("\n--------------------------------------")
        file.write("\n|              |   Count  | Percent  |")
        file.write("\n|--------------+----------+----------|")
        for pattern_name, pattern_count in pattern_count.items():
            if total_pattern_count != 0:
                percentage = int((pattern_count / total_pattern_count) * 100)
            else:
                percentage = 0
            file.write(f"\n| {pattern_name.ljust(max_pattern_length)} | {str(pattern_count).ljust(3)}" +
                       f"      | {str(percentage).ljust(3)}      |")
        file.write("\n|--------------+----------+----------|")
        file.write(f"\n| Total        | {str(total_pattern_count).ljust(3)}      |          |")
        file.write("\n--------------------------------------")


def update(frame_num, img, grid, width, height, output_file):
    global first_iteration
    if first_iteration:
        first_iteration = False
        return
    else:
        new_grid = check_neighbors(grid.copy(), grid.copy(), width, height)
        pattern_count = detect_patterns(new_grid, width, height)
        generate_iteration_log(frame_num, output_file, pattern_count)
        img.set_data(new_grid)
        grid[:] = new_grid[:]
        return img,


# main() function
def main():
    parser = ArgumentParser(description="Runs Conway's Game of Life system.py.")
    parser.add_argument('input_file', type=str, help='Input file for Game of Life')
    parser.add_argument('output_file', type=str, help='Output file with report log')

    args: Namespace = parser.parse_args()
    cell_array = []

    input_file = args.input_file
    output_file = args.output_file

    print(f"Simulation began at {datetime.now()}")
    with open(input_file, 'r') as file:
        grid_size = file.readline().rstrip().split(' ')
        width, height = int(grid_size[0]), int(grid_size[1])
        generations = int(file.readline().rstrip())
        for line in file:
            cell_coords = line.rstrip().split(' ')
            if cell_coords != ' ':
                cell_array.append([int(cell_coords[0]), int(cell_coords[1])])

    with open(output_file, 'a') as op_file:
        op_file.truncate(0)
        op_file.write(f"Simulation at {date.today()}")
        op_file.write(f"\nUniverse size: {width} x {height}")

    update_interval = 5

    if generations < 200:
        generations = 200

    grid = generate_grid(width, height, cell_array)

    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')

    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, width, height, output_file),
                                  frames=generations,
                                  interval=update_interval,
                                  repeat=False)

    plt.show()
    print(f"Simulation ended at {datetime.now()}")


# call main
if __name__ == '__main__':
    main()
