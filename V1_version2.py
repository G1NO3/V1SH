import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

R = 7
C = 7
A = 12
angle_bin = np.linspace(0,np.pi,13)[:-1]
angle_matrix = np.vstack((np.cos(angle_bin),np.sin(angle_bin)))
np.set_printoptions(precision=2,threshold=np.inf,linewidth=1000)
class NeuronGroup():
    def __init__(self,args:Dict):
        for k,v in args.items():
            setattr(self,k,v)
        self.X = np.zeros((R,C,A))
        self.Y = np.zeros((R,C,A))
        self.I = np.zeros((R,C,A))
        for m in range(R):
            for n in range(C):
                for t in range(A):
                    self.I[m][n][t] = self.I_ex[m][n][t] + (self.I_ex[m][n][(t+1)%A] + self.I_ex[m][n][(t-1)%A]) * np.exp(-8/12) \
                                    + (self.I_ex[m][n][(t+2)%A] + self.I_ex[m][n][(t-2)%A]) * np.exp(-8/6)
        for t in range(A):               
            print(self.I[...,t])   
        if 'JW.npy' in os.listdir():
            print('loading JW.npy')
            self.J, self.W = np.load('JW.npy')
        else:
            self.J = np.zeros((R,C,A,R,C,A))
            self.W = np.zeros((R,C,A,R,C,A))
            for m in range(R):
                for n in range(C):
                    for t in range(A):
                        self.J[m,n,t,...], self.W[m,n,t,...] = self.init_JW(m,n,t)
            np.save('JW.npy',[self.J,self.W])

        print('initialization completed')
        
        
    #activation function
    def gx(self,x):
        return np.minimum(np.maximum(0,x-self.Tx),1)
    def gy(self,y):
        return np.maximum(np.maximum(0,self.g1*y),self.g1 * self.Ly + self.g2 * (y - self.Ly))
    #one_step update
    def update(self):
        dX = np.zeros((R,C,A))
        dY = np.zeros((R,C,A))
        excitatory_output = self.gx(self.X)
        for m in range(R):
            for n in range(C):
                for t in range(A):
                    dX[m][n][t] = - self.psi_y(m,n,t) + np.sum(self.J[m][n][t]*excitatory_output) \
                         + self.I[m][n][t] + self.Io(m,n) + self.I_noise
                    dY[m][n][t] =  np.sum(self.W[m][n][t]*excitatory_output) + self.I_noise
        self.X += (-self.alpha_x * self.X - self.gy(self.Y) + self.Jo * excitatory_output + dX) * self.dt
        self.Y += (-self.alpha_y * self.Y + excitatory_output + self.Ic + dY ) * self.dt

    # inhibition within a hypercolumn
    def psi_y(self,m,n,t):
        return (self.Y[m][n][(t+1)%A] + self.Y[m][n][(t-1)%A]) * 0.8\
            + (self.Y[m][n][(t+2)%A] + self.Y[m][n][(t-2)%A]) * 0.7
    # calculate the normalized current(already checked)
    def Io(self,m,n):
        m_l,m_h,n_l,n_h = max(0,m-2),min(R-1,m+2),max(0,n-2),min(C-1,n+2)
        current_valid = self.X[m_l:m_h,n_l:n_h,:]
        normalized_current = -2.0 * (np.sum(current_valid)/((m_h-m_l+1)*(n_h-n_l+1)*A))**2
        if normalized_current<-10:
            print(normalized_current)
        return normalized_current + 0.85
    # calculate the distance between two neurons(already checked)
    def distance(self,m1,n1,m2,n2):
        return (np.sqrt((m1-m2)**2 + (n1-n2)**2))
    # calculate the angle between two neurons(including beta, theta1 and theta2, see the book)
    # from coordinate(pi/12) to angle: angle = coordinate*pi/12 (already checked)
    # notice: the return value of arccos is in [0,pi] not [0,180] 
    # def mid_angle(self,m1,n1,t1,m2,n2,t2,check=False):
        
    #     # vec_d = np.array([n2-n1,m2-m1])
    #     # vec_t1 = angle_matrix[:,t1]
    #     # vec_t2 = angle_matrix[:,t2]
    #     # delta_1 = np.arccos(np.dot(vec_d,vec_t1)/self.distance(m1,m2,n1,n2))
    #     # delta_1 = np.where(delta_1>np.pi/2,delta_1-np.pi,delta_1)
    #     # delta_2 = np.arccos(np.dot(vec_d,vec_t2)/self.distance(m1,m2,n1,n2))
    #     # delta_2 = np.where(delta_2>np.pi/2,delta_2-np.pi,delta_2)

    #     d_angle_0 = np.arccos((n2-n1)/self.distance(m1,m2,n1,n2))
    #     d_angle = np.fmin(d_angle_0, np.pi-d_angle_0)
    #     delta_1 = np.fabs(t1*np.pi/12-d_angle)
    #     delta_1 = np.fmin(delta_1, np.pi-delta_1)
    #     delta_2 = np.fabs(t2*np.pi/12-d_angle)
    #     delta_2 = np.fmin(delta_2,np.pi-delta_2)

    #     if check:
    #         print('distance')
    #         print(self.distance(m1,m2,n1,n2))
    #         print('d_angle')
    #         print(np.arccos((m2-m1)/self.distance(m1,m2,n1,n2)))
    #         print('delta_1')
    #         print(delta_1/np.pi*180)

    #         print('delta_2')
    #         print(delta_2/np.pi*180)

    #     return 2*np.fabs(delta_1) + 2*np.sin(np.fabs(delta_1+delta_2)), delta_1, delta_2, d_angle_0

    def angle_between_element_bar_and_connection_line(self, m1, n1, t1, m2, n2, t2):
        d_angle = np.arctan2(n2 - n1, m2 - m1)

        # delta_1 = np.fabs(t1-d_angle)
        # delta_1 = np.fmin(delta_1, 12.0-delta_1)
        # delta_2 = np.fabs(t2-d_angle)
        # delta_2 = np.fmin(delta_2,12.0-delta_2)

        delta_1 = t1*np.pi/12 - d_angle
        delta_2 = t2*np.pi/12 - d_angle

        if delta_1 > np.pi / 2:
            delta_1 -= np.pi
        elif delta_1 < -np.pi / 2:
            delta_1 += np.pi
        if delta_2 > np.pi / 2:
            delta_2 -= np.pi
        elif delta_2 < -np.pi / 2:
            delta_2 += np.pi

        return 2*np.fabs(delta_1) + 2*np.sin(np.fabs(delta_1+delta_2)), delta_1, delta_2
    #calculate the recurrent input(including J and W)
    def init_JW(self,m,n,t):
        # check=False
        # beta_matrix = np.zeros((R,C,A))
        # theta1_matrix = np.zeros((R,C,A))
        # theta2_matrix = np.zeros((R,C,A))

        J_connection_matrix = np.zeros((R,C,A))
        W_connection_matrix = np.zeros((R,C,A))
        # d_angle_matrix = np.zeros((R,C,A))
        recurrent_X = 0.0
        recurrent_Y = 0.0
        for m_prime in range(R):
            for n_prime in range(C):
                if m!=m_prime or n!=n_prime:
                    d = self.distance(m,n,m_prime,n_prime)
                    #calculate the recurrent excitation J
                    if d<=10:
                        for t_prime in range(A):
                            beta, theta_1, theta_2 = self.angle_between_element_bar_and_connection_line(m,n,t,m_prime,n_prime,t_prime)
                            # beta_matrix[m_prime][n_prime][t_prime] = beta
                            # theta1_matrix[m_prime][n_prime][t_prime] = theta_1/np.pi*180
                            # theta2_matrix[m_prime][n_prime][t_prime] = theta_2/np.pi*180
                            # d_angle_matrix[m_prime][n_prime][t_prime] = d_angle_0/np.pi*180
                            if beta<np.pi/2.69 or (beta<np.pi/1.1 and np.abs(theta_2)<np.pi/5.9):
                                # print('efficient J')
                                # print(m_prime,n_prime,t_prime)
                                J_connection_matrix[m_prime][n_prime][t_prime] = 0.126 * np.exp(-(beta/d)**2-2*(beta/d)**7-d**2/90)
                    
                    #calculate the recurrent inhibition W
                    if d>0:
                        for t_prime in range(A):
                            beta, theta_1, theta_2 = self.angle_between_element_bar_and_connection_line(m,n,t,m_prime,n_prime,t_prime)
                            delta_12 = np.abs(t-t_prime)*np.pi/12
                            delta_12 = np.fmin(delta_12, np.pi-delta_12)
                            print(delta_12)
                            if d/np.cos(beta/4)<10 and beta>=np.pi/1.1 and np.fabs(theta_1)>np.pi/11.999 and delta_12<np.pi/3:
                                W_connection_matrix[m_prime][n_prime][t_prime] = 0.141 * (1-np.exp(-0.4*(beta/d)**1.5)) * np.exp(-(delta_12/(np.pi/4))**1.5)
        # for t_prime in range(A):

        #     print(t_prime)
        # #     # print('d_angle')
        # #     # print(d_angle_matrix[...,t_prime])
        # #     print('beta')
        # #     print(beta_matrix[...,t_prime])
        # #     print('theta1')
        # #     print(theta1_matrix[...,t_prime])
        # #     print('theta2')
        # #     print(theta2_matrix[...,t_prime])

        #     print('connection')
        #     print(connection_matrix[...,t_prime])
        #     if t_prime!=0:
        #         print(np.all(connection_matrix[...,t_prime]==connection_matrix[::-1,::-1,12-t_prime]))
                

        # print(theta1_matrix[...,3]==theta1_matrix[...,9])
        return J_connection_matrix, W_connection_matrix
    def run(self,duration):
        step = round(duration/self.dt)
        for i in tqdm(range(step)):
            print('X')
            print(self.X[:,:,0])
            print('Y')
            print(self.Y[:,:,0])
            self.update()

        return self.X, self.Y


                            
                    
I_pattern = np.zeros((R,C,A))
I_pattern[R//2,C//2,0] = 1
I_pattern[...,6] = 1
V1 = NeuronGroup({
    'Tx':1,
    'Ly':1.2,
    'g1':0.21,
    'g2':2.5,
    'Ic':1,
    'Jo':0.8,
    'alpha_x':1,
    'alpha_y':1,
    'dt':0.01,
    'I_ex':I_pattern,
    'I_noise':0
})

# beta, theta1, theta2 = V1.JW(5,5,0)
# print(beta, theta1, theta2)
X,Y = V1.run(10)
print(X[:,:,0])
