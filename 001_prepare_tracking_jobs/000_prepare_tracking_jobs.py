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

print('\nEigenvalues:')
print(np.angle(np.linalg.eigvals(RR))/2/np.pi)


# Symplectify

def healy_symplectify(M):
    # https://accelconf.web.cern.ch/e06/PAPERS/WEPCH152.PDF
    print("Symplectifying linear One-Turn-Map...")

    print("Before symplectifying: det(M) = {}".format(np.linalg.det(M)))
    I = np.identity(6)

    S = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        ]
    )

    V = np.matmul(S, np.matmul(I - M, np.linalg.inv(I + M)))
    W = (V + V.T) / 2
    if np.linalg.det(I - np.matmul(S, W)) != 0:
        M_new = np.matmul(I + np.matmul(S, W),
                          np.linalg.inv(I - np.matmul(S, W)))
    else:
        print("WARNING: det(I - SW) = 0!")
        V_else = np.matmul(S, np.matmul(I + M, np.linalg.inv(I - M)))
        W_else = (V_else + V_else.T) / 2
        M_new = -np.matmul(
            I + np.matmul(S, W_else), np.linalg(I - np.matmul(S, W_else))
        )

    print("After symplectifying: det(M) = {}".format(np.linalg.det(M_new)))
    return M_new

RR_sym = healy_symplectify(RR)

import numpy as np

S = np.array([[0., 1., 0., 0., 0., 0.],
              [-1., 0., 0., 0., 0., 0.],
              [ 0., 0., 0., 1., 0., 0.],
              [ 0., 0.,-1., 0., 0., 0.],
              [ 0., 0., 0., 0., 0., 1.],
              [ 0., 0., 0., 0.,-1., 0.]])

######################################################
### Implement Normalization of fully coupled motion ##
######################################################

def Rot2D(mu):
    return np.array([[ np.cos(mu), np.sin(mu)],
                     [-np.sin(mu), np.cos(mu)]])

def _linear_normal_form(M):
    w0, v0 = np.linalg.eig(M)

    a0 = np.real(v0)
    b0 = np.imag(v0)

    index_list = [0,5,1,2,3,4]

    ##### Sort modes in pairs of conjugate modes #####

    conj_modes = np.zeros([3,2], dtype=np.int64)
    for j in [0,1]:
        conj_modes[j,0] = index_list[0]
        del index_list[0]

        min_index = 0
        min_diff = abs(np.imag(w0[conj_modes[j,0]] + w0[index_list[min_index]]))
        for i in range(1,len(index_list)):
            diff = abs(np.imag(w0[conj_modes[j,0]] + w0[index_list[i]]))
            if min_diff > diff:
                min_diff = diff
                min_index = i

        conj_modes[j,1] = index_list[min_index]
        del index_list[min_index]

    conj_modes[2,0] = index_list[0]
    conj_modes[2,1] = index_list[1]

    ##################################################
    #### Select mode from pairs with positive (real @ S @ imag) #####

    modes = np.empty(3, dtype=np.int64)
    for ii,ind in enumerate(conj_modes):
        if np.matmul(np.matmul(a0[:,ind[0]], S), b0[:,ind[0]]) > 0:
            modes[ii] = ind[0]
        else:
            modes[ii] = ind[1]

    ##################################################
    #### Sort modes such that (1,2,3) is close to (x,y,zeta) ####
    for i in [1,2]:
        if abs(v0[:,modes[0]])[0] < abs(v0[:,modes[i]])[0]:
            modes[0], modes[i] = modes[i], modes[0]

    if abs(v0[:,modes[1]])[2] < abs(v0[:,modes[2]])[2]:
        modes[2], modes[1] = modes[1], modes[2]

    ##################################################
    #### Rotate eigenvectors to the Courant-Snyder parameterization ####
    phase0 = np.log(v0[0,modes[0]]).imag
    phase1 = np.log(v0[2,modes[1]]).imag
    phase2 = np.log(v0[4,modes[2]]).imag

    v0[:,modes[0]] *= np.exp(-1.j*phase0)
    v0[:,modes[1]] *= np.exp(-1.j*phase1)
    v0[:,modes[2]] *= np.exp(-1.j*phase2)

    ##################################################
    #### Construct W #################################

    a1 = v0[:,modes[0]].real
    a2 = v0[:,modes[1]].real
    a3 = v0[:,modes[2]].real
    b1 = v0[:,modes[0]].imag
    b2 = v0[:,modes[1]].imag
    b3 = v0[:,modes[2]].imag

    n1 = 1./np.sqrt(np.matmul(np.matmul(a1, S), b1))
    n2 = 1./np.sqrt(np.matmul(np.matmul(a2, S), b2))
    n3 = 1./np.sqrt(np.matmul(np.matmul(a3, S), b3))

    a1 *= n1
    a2 *= n2
    a3 *= n3

    b1 *= n1
    b2 *= n2
    b3 *= n3

    W = np.array([a1,b1,a2,b2,a3,b3]).T
    W[abs(W) < 1.e-14] = 0. # Set very small numbers to zero.
    invW = np.matmul(np.matmul(S.T, W.T), S)

    ##################################################
    #### Get tunes and rotation matrix in the normalized coordinates ####

    mu1 = np.log(w0[modes[0]]).imag
    mu2 = np.log(w0[modes[1]]).imag
    mu3 = np.log(w0[modes[2]]).imag

    #q1 = mu1/(2.*np.pi)
    #q2 = mu2/(2.*np.pi)
    #q3 = mu3/(2.*np.pi)

    R = np.zeros_like(W)
    R[0:2,0:2] = Rot2D(mu1)
    R[2:4,2:4] = Rot2D(mu2)
    R[4:6,4:6] = Rot2D(mu3)
    ##################################################

    return W, invW, R

RR_sym = healy_symplectify(RR)
W, invW, Rotation = _linear_normal_form(RR_sym)

ampl_sigmas = 1.
norm_emit_x = 2.5e-6
geom_emit_x = norm_emit_x / particles.beta0 / particles.gamma0

n_part = 100
theta = np.linspace(0, 2*np.pi, n_part)
x_norm = ampl_sigmas * np.sqrt(geom_emit_x) * np.cos(theta)
px_norm = ampl_sigmas * np.sqrt(geom_emit_x) * np.sin(theta)

XX_norm= np.array([x_norm,
                   px_norm,
                   np.zeros(n_part),
                   np.zeros(n_part),
                   np.zeros(n_part),
                   np.zeros(n_part)])

XX = np.dot(W, XX_norm)

particles_matched = xt.Particles(
        p0c=p0c,
        x=XX[0, :], px=XX[1, :], y=XX[2, :], py=XX[3, :],
        zeta=XX[4, :], delta=XX[5, :])


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(particles_matched.x, particles_matched.px)

tracker.track(particles_matched, num_turns=10)

plt.plot(particles_matched.x, particles_matched.px)

plt.show()
