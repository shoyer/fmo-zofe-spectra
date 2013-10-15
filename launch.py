from task_runner import run_task
from fmo_spectra import calculate_spectra


ALL_PARAMETERS = [
    {
        'sample_id': sample_id,
        'random_seed': 424242,
        'xopt': xopt,
        'pop_times': pop_times,
        'ode_settings': {
            'rtol': 1e-6,
            'max_step': 50,
            'nsteps': 1e5
        }
    }
    for xopt, pop_times in zip(
        [[],
         [45.8129458572, 114.744881171, -58.3097813804,
          -480.241126371, -96.1162627658, 821.250432297,
          459.018733158, -370.744505636, -258.474007389]],
        [[0, 75, 150, 300, 600, 1200, 10000], [0]])
    for sample_id in range(1000)
]


if __name__ == '__main__':
    run_task(calculate_spectra, ALL_PARAMETERS)
