import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

R = 5
C = 5
A = 12
angle_bin = np.linspace(0,180,13)[:-1]
class NeuronGroup():
    def __init__(self,args:Dict):
        for k,v in args.items():
            setattr(self,k,v)
        self.X = np.zeros((R,C,A))
        self.Y = np.zeros((R,C,A))

    def gx(self,x):
        return np.min(np.max(0,x-self.Tx),1)
    def gy(self,y):
        return np.max(np.max(0,self.g1*y),self.g1 * self.Ly + self.g2 * (y - self.Ly))
    def update(self):
        dX = np.zeros((R,C,A))
        dY = np.zeros((R,C,A))
        for m in range(R):
            for n in range(C):
                for t in range(A):
                    recurrent_X, recurrent_Y = self.JW(m,n,t)
                    dX[m][n][t] = - self.psi_y(m,n,t) + recurrent_X \
                         + self.I(m,n,t) + self.Io(m,n) + self.I_noise
                    dY[m][n][t] =  recurrent_Y + self.I_noise
        self.X += (-self.alpha_x * self.X - self.gy(self.Y) + self.Jo * self.gx(self.X) + dX) * self.dt
        self.Y += (-self.alpha_y * self.Y + self.gx(self.X) + self.Ic + dY ) * self.dt

    def psi_y(self,m,n,t):
        return (self.Y[m][n][(t+1)%A] + self.Y[m][n][(t-1)%A]) * 0.8\
            + (self.Y[m][n][(t+2)%A] + self.Y[m][n][(t-2)%A]) * 0.7
    def I(self,m,n,t):
        return self.I_ex[m][n][t] + (self.I_ex[m][n][(t+1)%A] + self.I_ex[m][n][(t-1)%A]) * np.exp(-8/12) \
            + (self.I_ex[m][n][(t+2)%A] + self.I_ex[m][n][(t-2)%A]) * np.exp(-8/6)
    def Io(self,m,n):
        m_l,m_h,n_l,n_h = max(0,m-2),min(R-1,m+2),max(0,n-2),min(C-1,n+2)
        current_valid = self.X[m_l:m_h,n_l:n_h,:]
        normalized_current = -2.0 * (np.sum(current_valid)/(np.sum(np.ones_like(current_valid)/A)))**2
        return normalized_current + 0.85
    def JW(self,m,n,t):
        def distance(m1,m2,n1,n2):
            return np.abs(m1-m2) + np.abs(n1-n2)
        def mid_angle(m1,n1,t1,m2,n2,t2):
            d_angle = np.arctan((m2-m1)/(n2-n1))/np.pi * 12
            if d_angle<0:
                d_angle = 12+d_angle
            delta_1 = np.fabs(t1-d_angle)
            delta_1 = np.fmin(delta_1, 12.0-delta_1)
            delta_2 = np.fabs(t2-d_angle)
            delta_2 = np.fmin(delta_2,12.0-delta_2)

            return 2*delta_1*np.pi/12 + 2*np.sin((delta_1+delta_2)*np.pi/12), delta_1*np.pi/12, delta_2*np.pi/12

        recurrent_X = 0.0
        recurrent_Y = 0.0
        for m_prime in range(R):
            if m!=m_prime:
                for n_prime in range(C):
                    if n!=n_prime:
                        d = distance(m,n,m_prime,n_prime)
                        if d>0 and d<=10:
                            for t_prime in range(A):
                                if t!=t_prime:
                                    beta, theta_1, theta_2 = mid_angle(m,n,t,m_prime,n_prime,t_prime)
                                    if beta<np.pi/2.69 or (beta<np.pi/1.1 and theta_2<np.pi/5.9):
                                        recurrent_X += 0.126 * np.exp(-(beta/d)**2-2*(beta/d)**7-d**2/90)
                        if d>0:
                            for t_prime in range(A):
                                if t!=t_prime:
                                    beta, theta_1, theta_2 = mid_angle(m,n,t,m_prime,n_prime,t_prime)
                                    delta_12 = np.abs(t-t_prime)*np.pi/12
                                    delta_12 = np.fmin(delta_12, np.pi-delta_12)
                                    if d/np.cos(beta/4)<=10 and beta>=np.pi/1.1 and theta_1>np.pi/11.999 and delta_12<np.pi/3:
                                        recurrent_Y += 0.141 * (1-np.exp(-0.4*(beta/d)**1.5)) * np.exp(-(delta_12/(np.pi/4))**1.5)
        return recurrent_X, recurrent_Y
    def run(self,duration):
        step = round(duration/self.dt)
        for i in tqdm(range(step)):
            self.update()
            print(self.X[:,:,0])
            print(self.Y[:,:,0])
        return self.X, self.Y


                            
                    
I_pattern = np.zeros((R,C,A))
I_pattern[:,:,0] = 1
V1 = NeuronGroup({
    'Tx':1,
    'Ly':1.2,
    'g1':0.21,
    'g2':2.5,
    'Ic':1,
    'Jo':0.8,
    'alpha_x':1,
    'alpha_y':1,
    'dt':0.1,
    'I_ex':I_pattern,
    'I_noise':0
})

X,Y = V1.run(10)
print(X[:,:,0])
