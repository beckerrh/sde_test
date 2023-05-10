import numpy as np
import matplotlib.pyplot as plt
import pathlib

#------------------------------------------------------------------
class LorenzRun():
    def __init__(self, sigma=10, rho=28, beta=8/3):
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.A = np.array(((sigma, -sigma, 0),(sigma, 1, 0),(0,0,beta)))
        self.l = np.array((0,0,-beta*(rho+sigma)))
#----------------------------------------------------------
    def _run_cnse(self, u, dt, nt, q, noise):
        B = np.eye(3) + 0.5 * dt * self.A
        for it in range(1, nt):
            uold = u[it - 1]
            B[1, 2] = 0.5 * dt * uold[0]
            B[2, 1] = -0.5 * dt * uold[0]
            id = 1-0.5*np.sqrt(dt)*q*noise[it]
            B[0, 0] = id + dt*self.A[0,0]
            B[1, 1] = id + dt*self.A[1,1]
            B[2, 2] = id + dt*self.A[2,2]
            b = 2 * uold + dt * self.l
            u[it] = np.linalg.solve(B, b) - uold
    def _run_cnsi(self, u, dt, nt, q, noise):
        B = np.eye(3) + 0.5 * dt * self.A
        for it in range(1, nt):
            uold = u[it - 1]
            B[1, 2] = 0.5 * dt * uold[0]
            B[2, 1] = -0.5 * dt * uold[0]
            B[1, 0] = 0.5 * dt * (self.sigma + uold[2])
            B[2, 0] = -0.5 * dt * uold[1]
            b = 2 * uold + dt * self.l
            b[1] += dt * uold[0] * uold[2]
            b[2] -= dt * uold[0] * uold[1]
            C = B - np.sqrt(dt)*q*noise[it]*np.eye(3)
            u[it] = np.linalg.solve(C, b) - uold
    def _run_ie(self, u, dt, nt, q, noise):
        B = (1-0.5*q**2*dt)*np.eye(3) + dt * self.A
        for it in range(1, nt):
            uold = u[it - 1]
            b = uold + dt * self.l + q*np.sqrt(dt)*noise[it-1]*uold
            B[1, 2] = dt * uold[0]
            B[2, 1] = -dt * uold[0]
            u[it] = np.linalg.solve(B, b)
    def _run_ee(self, u, dt, nt, q, noise):
        B = np.eye(3) - dt * self.A
        for it in range(1, nt):
            uold = u[it - 1]
            u[it] = B @ uold + dt * self.l
            u[it, 1] -= dt * uold[0] * uold[2]
            u[it, 2] += dt * uold[0] * uold[1]
