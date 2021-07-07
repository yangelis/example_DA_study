import pickle

import numpy as np
from scipy.optimize import fsolve

import xline as xl
import xtrack as xt

with open('../000_machine_model/optics_orbit_at_start_ring.pkl', 'rb') as fid:
    ddd = pickle.load(fid)
p0c = ddd['p0c_eV']

line = xl.Line.from_json(
        '../000_machine_model/xline/line_bb_dipole_not_cancelled.json')

line_bb_off = line.copy()
line_bb_off.disable_beambeam()

tracker = xt.Tracker(sequence=line_bb_off)

def one_turn_map(p):
    # TODO  To be generalized for ions
    part = xt.Particles(
            p0c = p0c,
            x = p[0],
            px = p[1],
            y = p[2],
            py = p[3],
            zeta = p[4],
            delta = p[5])
    tracker.track(part)
    p_res = np.array([
           part.x[0],
           part.px[0],
           part.y[0],
           part.py[0],
           part.zeta[0],
           part.delta[0]])
    return p_res


print('Start CO search')
res = fsolve(lambda p: p - one_turn_map(p), x0=np.array(6*[0]))
print('Done CO search')

part_co_dict = {'p0c': p0c, 'x': res[0], 'px': res[1], 'y': res[2], 'py': res[3],
                'zeta': res[4], 'delta': res[5]}

particles = xt.Particles(**part_co_dict)
print('Test closed orbit')
for _ in range(10):
    print(particles.at_turn[0], particles.x[0], particles.y[0],
          particles.zeta[0])
    tracker.track(particles)

# Find R matrix
p0 = res.copy()
II = np.eye(6)
RR = np.zeros((6, 6), dtype=np.float64)
for jj,dd in enumerate([1e-9,1e-12,1e-9,1e-12,1e-9,1e-9]):
    pd=p0+II[jj]*dd
    RR[:,jj]=(one_turn_map(pd)-p0)/dd

