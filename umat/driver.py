
def driver(F_final, n_steps, orientation):

    f = lambda t: F_init + t * (F_final - F_init)

    times = np.linspace(0, 1, num=n_steps + 1)
    F_init = torch.eye(3)

    for t0, t1 in zip(times[:-1], times[1:]):
        F0, F1 = f(t0), f(t1)
