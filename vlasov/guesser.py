import numpy as np
def make_guess(theta, solver):
    diff_old = 3*np.pi/180
    gamold = 0.0
    guess_made = False
    for m in range(solver.M):
        diff = abs(theta - solver.solutions[m]['theta'])
        gam = np.max(solver.solutions[m]['omega'].imag)
        if diff <= diff_old and gam > gamold:
            m1 = m
            gamold = gam
            diff_old = diff
            guess_made = True
    if guess_made:
        index = np.argmax(solver.solutions[m1]['omega'].imag)
        k0 = solver.solutions[m1]['k'][index]
        guess = solver.solutions[m1]['omega'][index]
        return k0, guess
    else:
        return None, None