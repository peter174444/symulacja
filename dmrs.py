import matplotlib.pyplot as plt
import numpy as np


def make_global_grid(data_symbols, num_rb=5, pilot_symbols=[2, 11], pilot_val=1.0+1j):
    grid = np.zeros((14, num_rb * 12), dtype=complex)
    data_idx = 0

    for s in range(14):
        if s in pilot_symbols:
            # cały symbol pilotowy: pilot co 2 podnośne w każdym RB
            for rb in range(num_rb):
                for k in range(0, 12, 2):
                    sc = rb * 12 + k
                    grid[s, sc] = pilot_val
            # reszta w tym symbolu zostaje 0
        else:
            # dane: w tym symbolu wypełniamy po kolei RB0..RB4
            for rb in range(num_rb):
                sc_start = rb * 12
                grid[s, sc_start:sc_start+12] = data_symbols[data_idx:data_idx+12]
                data_idx += 12

    return grid

def visualize_grid(grid):
    color = np.zeros(grid.shape, dtype=int)

    # piloty = czerwone
    color[grid == (1.0 + 1j)] = 2

    # dane = niebieskie
    color[(grid != 0) & (grid != (1.0 + 1j))] = 1

    # zera = białe

    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    cmap = ListedColormap(["white", "blue", "red"])

    plt.figure(figsize=(12, 6))
    plt.imshow(color.T, aspect='auto', cmap=cmap, origin='lower')
    plt.xticks(np.arange(grid.shape[0]))
    plt.xlabel("Symbol OFDM (czas)")
    plt.ylabel("Podnośna (częstotliwość)")
    plt.title("Globalna siatka OFDM (5 RB)")

    # legenda
    legend_patches = [
        mpatches.Patch(color="white", label="Zero (puste RE)"),
        mpatches.Patch(color="blue", label="Dane QAM"),
        mpatches.Patch(color="red", label="Pilot DMRS"),
    ]
    plt.legend(handles=legend_patches, loc="upper right")

    plt.show()
