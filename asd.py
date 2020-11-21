import numpy as np

shape = (4, 4)
max_steps = shape[0] * shape[1]
desc = np.zeros(shape)
desc[shape[0] - 1, 0] = 1
desc[shape[0] - 1, shape[1] - 1] = 1

nA = 4
nS = 4 * 4
nrow = 4
ncol = 4
# isd = np.array(desc == b'S').astype('float64').ravel()
# isd /= isd.sum()

P = {s: {a: [] for a in range(nA)} for s in range(nS)}


def to_s(row, col):
    return row * 4 + col


def inc(row, col, a):
    if a == 0:
        col = max(col - 1, 0)
    elif a == 1:
        row = min(row + 1, nrow - 1)
    elif a == 2:
        col = min(col + 1, ncol - 1)
    elif a == 3:
        row = max(row - 1, 0)
    return (row, col)

def update_probability_matrix(row, col, action):
    newrow, newcol = inc(row, col, action)
    newstate = to_s(newrow, newcol)
    newletter = desc[newrow, newcol]
    done = bytes(newletter) in b'GH'
    reward = float(newletter == b'G')
    return newstate, reward, done

for row in range(4):
    for col in range(4):
        s = to_s(row, col)
        for a in range(4):
            li = P[s][a]
            letter = desc[row, col]
            if letter in b'GH':
                li.append((1.0, s, 0, True))
            else:
                for b in [(a - 1) % 4, a, (a + 1) % 4]:
                    li.append((
                        1. / 3.,
                        *update_probability_matrix(row, col, b)
                    ))