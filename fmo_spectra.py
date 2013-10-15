import logging
import numpy as np
import pandas as pd
from qspectra import (CM_FS, CM_K, ZOFEModel, bound_signal,
                      ElectronicHamiltonian, PseudomodeBath, DebyeBath,
                      RedfieldModel, simulate_pump, impulsive_probe)

from pulse import poly_pulse_scaled


electronic_fmo = np.array(np.mat("""
    12400 -87.7 5.5 -5.9 6.7 -13.7 -9.9;
    -87.7 12520 30.8 8.2 0.7 11.8 4.3;
    5.5 30.8 12200 -53.5 -2.2 -9.6 6.;
    -5.9 8.2 -53.5 12310 -70.7 -17. -63.3;
    6.7 0.7 -2.2 -70.7 12470 81.1 -1.3;
    -13.7 11.8 -9.6 -17. 81.1 12620 39.7;
    -9.9 4.3 6. -63.3 -1.3 39.7 12430
    """))

dipoles_fmo = np.array([d / np.linalg.norm(d) for d in
    np.array([[3.019, 3.442, 0.797, 3.213, 2.969, 0.547, 1.983],
              [2.284, -2.023, -3.871, 2.145, -2.642, 3.562, 2.837],
              [1.506, 0.431, 0.853, 1.112, -0.661, -1.851, 2.015]]).T])

# Bath parameters for pseudomode bath -- fit to the Drude spectral density
# for FMO for 77K of Ishizaki and Fleming (each PM is represented by a
# Lorentzian at frequency Omega, with width gamma, and of strength huang in
# the bath correlation SPECTRUM, NOT spectral density)
Omega = [-500., -200., -90., 1., 21., 60., 80., 130., 200., 300., 400.,
         500., 600., 800., 1100., 1500.] # frequencies of PMs
huang = [-2.5133e-03, -7.5398e-03, -2.5133e-02, 5.0265e+01, 2.2619e+00,
          4.5239e-02, 2.7646e-01, 9.2991e-03, 2.2619e-02, 1.5080e-02,
          3.0159e-03, 3.5186e-03, 2.8274e-04, 1.7593e-03, 4.3982e-04,
          4.3982e-04] # Huang-Rhys factors of PMs (couplings to PMs)
gamma = [500., 100., 50., 50., 50., 50., 80., 40., 80., 150., 200., 200.,
         80., 250., 200., 300.] # dampings of the PMs
n_sites = len(electronic_fmo)
numb_pm = len(Omega)
on = np.ones(n_sites, complex)

Omega = np.array([Omega[pm]*on for pm in range(numb_pm)])
huang = np.array([huang[pm]*on for pm in range(numb_pm)])
gamma = np.array([gamma[pm]*on for pm in range(numb_pm)])


def fmo_zofe(sample_id, random_seed):
    """Returns the appropriate ZOFE dynamical model"""
    orig_hamiltonian = ElectronicHamiltonian(electronic_fmo, disorder_fwhm=100,
        bath=PseudomodeBath(numb_pm, Omega, gamma, huang), dipoles=dipoles_fmo)
    # run through the ensemble Hamiltonians until reaching member sample_id
    for ham in orig_hamiltonian.sample_ensemble(sample_id + 1,
            random_seed=random_seed):
        pass
    return ZOFEModel(ham, hilbert_subspace='gef', unit_convert=CM_FS)


def fmo_redfield(sample_id, random_seed):
    """Returns the equivalent Redfield dynamical model (for debugging)"""
    ham = fmo_zofe(sample_id, random_seed).hamiltonian
    ham.bath = DebyeBath(CM_K * 77, 35, 106)
    return RedfieldModel(ham, hilbert_subspace='gef', unit_convert=CM_FS)


def calculate_spectra(task_root, debug, sample_id, random_seed, xopt, pop_times,
                      ode_settings):
    logging.info('setting up control pulse and dynamics')
    make_dynamics = fmo_redfield if debug else fmo_zofe
    dynamical_model = make_dynamics(sample_id, random_seed)
    pump = poly_pulse_scaled(xopt)

    logging.info('simulating pump')
    t, state_vec = simulate_pump(dynamical_model, pump, 'x', times=pop_times,
                                 exact_isotropic_average=True, **ode_settings)

    logging.info('calculating spectra')
    f, spectra = impulsive_probe(dynamical_model, state_vec, 5000, 'xx',
                                 exact_isotropic_average=True,
                                 heisenberg_picture=False, **ode_settings)
    f, spectra = bound_signal(f, spectra, [12000, 12800], axis=-1)

    logging.info('saving spectra to HDF5')
    with pd.io.pytables.get_store(task_root + '.h5', mode='w', complevel=9,
                                  complib='zlib') as store:
        store['spectra'] = pd.DataFrame(spectra.real, t, f)
    return {'completed': True}
