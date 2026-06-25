import matplotlib.pyplot as plt
import numpy as np

def make_rb(data_symbols, pilot_symbols=[0, 12]):
    rb = np.zeros((14, 12), dtype=complex)

    # 1) symbole pilotowe
    for s in pilot_symbols:
        for k in range(0, 12, 2):   # co druga podnośna
            rb[s, k] = 1.0 + 1j     # pilot
        # k+1 zostaje 0

    # 2) pozostałe symbole = dane
    data_idx = 0
    for s in range(14):
        if s not in pilot_symbols:
            rb[s] = data_symbols[data_idx:data_idx+12]
            data_idx += 12

    return rb

# def make_global_grid(data_symbols, num_rb=5):
#     rbs = []
#     offset = 0

#     for _ in range(num_rb):
#         rb_data = data_symbols[offset : offset + 12*12]  # 12 symboli danych × 12 podnośnych
#         offset += 12*12

#         rb = make_rb(rb_data, pilot_symbols=[2, 11])
#         rbs.append(rb)

#     grid = np.hstack(rbs)   # 14 × (num_rb*12)
#     return grid

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


def visualize_rb(rb):
    color = np.zeros(rb.shape, dtype=int)

    # piloty = czerwone (dokładnie wartość 1+1j lub 1.5+1j)
    color[rb == (1.0 + 1j)] = 2

    # dane = niebieskie (cokolwiek niezerowego, co nie jest pilotem)
    color[(rb != 0) & (rb != (1.0 + 1j))] = 1

    # zera = 0 → białe

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", "blue", "red"])

    plt.figure(figsize=(5, 6))
    plt.imshow(color.T, aspect='auto', cmap=cmap, origin='lower')
    plt.xlabel("Symbol OFDM (czas)")
    plt.ylabel("Podnośna (częstotliwość)")
    plt.colorbar(ticks=[0,1,2], label="Typ RE")
    plt.title("Pojedynczy Resource Block (12×14)")
    plt.show()


def visualize_grid(grid):
    color = np.zeros(grid.shape, dtype=int)

    # piloty = czerwone
    color[grid == (1.0 + 1j)] = 2

    # dane = niebieskie
    color[(grid != 0) & (grid != (1.0 + 1j))] = 1

    # zera = białe

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", "blue", "red"])

    plt.figure(figsize=(12, 6))
    plt.imshow(color.T, aspect='auto', cmap=cmap, origin='lower')
    plt.xlabel("Symbol OFDM (czas)")
    plt.ylabel("Podnośna (częstotliwość)")
    plt.colorbar(ticks=[0,1,2], label="Typ RE")
    plt.title("Globalna siatka OFDM (5 RB)")
    plt.show()
