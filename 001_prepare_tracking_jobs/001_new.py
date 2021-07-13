import xline as xl
import xtrack as xt
import json
import help_functions as hf

with open('/home/HPC/sterbini/DA_study_example/000_machine_model/xlines/line_bb_for_tracking.json') as fid:
    dd=json.load(fid)

WW = dd['WW_finite_diffs']
invWW = dd['WWInv_finite_diffs']
closed_orbit = dd['particle_on_tracker_co']

p_co = xl.Particles.from_dict(dd['particle_on_tracker_co'])
line = xl.Line.from_dict(dd)

epsn_1  = 2.5e-6  # normalised emittance
epsn_2  = 2.5e-6
n_turns = 100
n_sigma = 10
ptau_max = 0.0
r_N = n_sigma*15
theta_N = 5

egeom_1 = epsn_1 / p_co.gamma0 / p_co.beta0
egeom_2 = epsn_2 / p_co.gamma0 / p_co.beta0

init_canonical_6D, A1_A2_in_sigma, number_of_particles = hf.get_DA_distribution(
    n_sigma=n_sigma,
    ptau_max=ptau_max,
    egeom_1=egeom_1,
    egeom_2=egeom_2,
    r_N=r_N,
    theta_N=theta_N,
    W=WW,
    invW=invWW,
)

pp = hf.add_to_closed_orbit(init_canonical_6D, p_co)

line.remove_inactive_multipoles(inplace=True)
line.remove_zero_length_drifts(inplace=True)
line.merge_consecutive_drifts(inplace=True)
line.merge_consecutive_multipoles(inplace=True)

tracker = xt.Tracker(sequence=line)
particles = xt.Particles(**pp.to_dict())

particles.num_particles = 30
n_turns = 200

import time
a=time.time()
tracker.track(particles, turn_by_turn_monitor=True, num_turns=n_turns)
b=time.time()  

print(f'Elapsed time: {b-a} s')
print(f'Elapsed time per particle per turn: {(b-a)/particles.num_particles/n_turns*1e6} us')

#tracker.record_last_track[0, :]
