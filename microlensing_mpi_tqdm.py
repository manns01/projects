#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:41:04 2023

@author: manishtamta
"""

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad
from tqdm import tqdm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

def integrand(s, l, n):
    return s * ((s**2 + l**2)**(n/2))

def normalization(w, n):
    return w**(2 + n)

def calculate_m(t, tm, n):
    s_low_lim = 0
    s_up_lim = abs(t)
    l_low_lim = 0
    l_up_lim = lambda s: np.sqrt(abs(tm**2 - s**2))

    I, _ = dblquad(integrand, s_low_lim, s_up_lim, lambda x: l_low_lim, l_up_lim, args=(n,))
    Inorm, _ = quad(normalization, 0, tm, args=(n,))

    if abs(t) < tm:
        m = I / Inorm
    else:
        m = 1
    return m

def magnification(t, m, dmdt):
    mu = (1/(abs(1 - (m/t**2))))*(1/(abs(1 + (m/t**2) - (dmdt/t))))
    return mu

def solveq(coeff):
    a, b, c = coeff
    D = np.sqrt(b**2 - 4*a*c)
    root1 = (-b + D) / (2*a)
    root2 = (-b - D) / (2*a)
    return root1, root2

def main():
    num = 100
    t_list = np.linspace(-2, 2, num)
    tm_list = np.arange(0.5, 100, 5)
    tm_arr = np.array(tm_list)
    local_tm_values = np.array_split(tm_arr, nproc)[rank]

    local_u1p34 = []

    for tm in tqdm(local_tm_values, desc=f'Process {rank}', position=rank):
        m_list = []

        for t in t_list:
            m = calculate_m(t, tm, -9/4)
            m_list.append(m)

        dmdt_list = np.gradient(m_list, t_list)

        u_list = [t_list[j] - m_list[j]/t_list[j] for j in range(num)]

        mu_list = [magnification(t_list[j], m_list[j], dmdt_list[j]) for j in range(num)]

        u_arr = np.array(u_list)
        t_arr = np.array(t_list)
        u_plus = u_arr[u_arr > 0]
        positive_indices = np.where(u_arr > 0)[0]
        t_plus = [t_arr[i] for i in positive_indices]

        mu_plus = [mu_list[i] for i in positive_indices]
        m_plus = [m_list[k] for k in positive_indices]
        dmdt_plus = [dmdt_list[l] for l in positive_indices]

        u1p34_list = []

        for k in range(len(mu_plus)):
            u = u_plus[k]
            m = m_plus[k]
            dmdt = dmdt_plus[k]
            coefficients = [1, -u, -m]
            roots = solveq(coefficients)
            t1 = roots[0]
            t2 = roots[1]
            mu1 = magnification(t1, m, dmdt)
            mu2 = magnification(t2, m, dmdt)
            mu_total = mu1 + mu2
            if mu_total > 1.34:
                u1p34_ = u
                u1p34_list.append(u1p34_)

        u1p34_max = max(u1p34_list)
        local_u1p34.append(u1p34_max)

    global_u1p34 = comm.gather(local_u1p34, root=0)

    if rank == 0:
        final_u1p34 = [item for sublist in global_u1p34 for item in sublist]
        print(final_u1p34)
    
    return tm_list, final_u1p34

if __name__ == "__main__":
    tm_list, final_u1p34 = main()

# Plotting
plt.plot(tm_list, final_u1p34, '-', label=r'$u_{1.34}$')
plt.xlabel(r'$t_m$')
plt.ylabel(r'$u_{1.34}$')
plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.grid()
plt.show()
