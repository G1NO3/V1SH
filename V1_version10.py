import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import turtle
R = 10
C = 10
A = 12
angle_bin = np.linspace(0,np.pi,13)[:-1]
angle_matrix = np.vstack((np.cos(angle_bin),np.sin(angle_bin)))
np.set_printoptions(precision=2,threshold=np.inf,linewidth=1000)
class NeuronGroup():
    def __init__(self,args:Dict):
        for k,v in args.items():
            setattr(self,k,v)
        self.X = np.random.normal(0,1,(R,C,A))
        self.Y = np.random.normal(0,1,(R,C,A))
        self.I = np.zeros((R,C,A))
        for m in range(R):
            for n in range(C):
                for t in range(A):
                    self.I[m][n][t] = self.I_ex[m][n][t] + (self.I_ex[m][n][(t+1)%A] + self.I_ex[m][n][(t-1)%A]) * np.exp(-8/12) \
                                    + (self.I_ex[m][n][(t+2)%A] + self.I_ex[m][n][(t-2)%A]) * np.exp(-8/6)
        # for t in range(A):               
        #     print(self.I[...,t])   
        name = 'JW'+str(R)+str(C)+'.npy'
        if name in os.listdir():
            print('loading '+name)
            self.J, self.W = np.load(name)
        else:
            print('initializing JW.npy')
            self.J = np.zeros((R,C,A,R,C,A))
            self.W = np.zeros((R,C,A,R,C,A))
            for m in tqdm(range(R)):
                for n in range(C):
                    for t in range(A):
                        self.J[m,n,t,...], self.W[m,n,t,...] = self.init_JW(m,n,t)
            np.save(name,[self.J,self.W])

        print('initialization completed')
        
    def check_J(self, position):
        m,n,t = position
        for angle in range(A):
            print(self.J[m,n,t,...,angle])
    def check_W(self, position):
        m,n,t = position
        for angle in range(A):
            print(self.W[m,n,t,...,angle])
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
                    dX[m][n][t] = - 0.1 * self.psi_y(m,n,t) + np.sum(self.J[m][n][t]*excitatory_output) \
                         + self.I[m][n][t] + self.Io(m,n) + self.I_noise
                    dY[m][n][t] =  np.sum(self.W[m][n][t]*excitatory_output) + self.I_noise
        self.X += (-self.alpha_x * self.X - self.gy(self.Y) + self.Jo * excitatory_output + dX) * self.dt
        self.Y += (-self.alpha_y * self.Y + excitatory_output + self.Ic + dY ) * self.dt

    # inhibition within a hypercolumn
    def psi_y(self,m,n,t):
        m_l,m_h,n_l,n_h= max(0,m-2),min(R-1,m+2),max(0,n-2),min(C-1,n+2)
        return np.sum(self.I_ex[m_l:m_h,n_l:n_h,t]) +\
             0.8 * (np.sum(self.I_ex[m_l:m_h,n_l:n_h,(t+1)%A]) + np.sum(self.I_ex[m_l:m_h,n_l:n_h,(t-1)%A])) +\
                0.7 * (np.sum(self.I_ex[m_l:m_h,n_l:n_h,(t+2)%A])) + np.sum(self.I_ex[m_l:m_h,n_l:n_h,(t-2)%A])
    
    # calculate the normalized current(already checked)
    def Io(self,m,n):
        m_l,m_h,n_l,n_h = max(0,m-2),min(R-1,m+2),max(0,n-2),min(C-1,n+2)
        current_valid = self.X[m_l:m_h,n_l:n_h,:]
        normalized_current = - 2.0 * (np.sum(current_valid)-np.sum(self.X[m,n,:]))/(((m_h-m_l+1)*(n_h-n_l+1)*A)-1)**2


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
                            # print(delta_12)
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
            # print('X')
            # print(self.X[:,:,0])
            # print('Y')
            # print(self.Y[:,:,0])
            self.update()

        return self.X, self.Y
    
    def zscore(self,ROI):
        r, c, a = ROI
        return (self.X[r,c,a] - self.X.mean())/self.X.std()

    # 11-horizontoal 6-vertical
    def init_input_pattern(args):
        if args.index==0:
            # single_search
            I_pattern =  np.zeros((R,C,A))
            I_pattern[:,:,A//2] = 1
            I_pattern[R//2,C//2,A//2] = 0
            I_pattern[R//2,C//2,args.center_angle] = 1
            return I_pattern
        elif args.index==1:
            # existing_absent_feature_search
            I_pattern_1 = np.zeros((R,C,A))
            I_pattern_1[:,:,A//2] = 1
            I_pattern_1[R//2,C//2,0] = 1
            I_pattern_2 =  np.zeros((R,C,A))
            I_pattern_2[:,:,A//2] = 1
            I_pattern_2[:,:,0] = 1
            I_pattern_2[R//2,C//2,A//2] = 0
            return I_pattern_1, I_pattern_2
        elif args.index==2:
            # relevant_irrelevant_feature_search
            I_relevant = np.zeros((R,C,A))  
            I_relevant[:,:C//2,A//4*3] = 1
            I_relevant[:,C//2:,A//4] = 1
            I_irrelevant = np.zeros((R,C,A))
            R_odd_index,R_even_index = np.arange(0,R,2),np.arange(1,R,2)
            C_odd_index,C_even_index = np.arange(0,C,2),np.arange(1,C,2)
            index_1 = np.meshgrid(R_odd_index, C_odd_index)
            index_2 = np.meshgrid(R_even_index, C_even_index)
            index_3 = np.meshgrid(R_odd_index, C_even_index)
            index_4 = np.meshgrid(R_even_index, C_odd_index)
            I_irrelevant[index_1[0], index_1[1],A//2] = 1
            I_irrelevant[index_2[0], index_2[1],A//2] = 1
            I_irrelevant[index_3[0], index_3[1],0] = 1
            I_irrelevant[index_4[0], index_4[1],0] = 1
            I_conjunction = I_irrelevant + I_relevant
            return I_relevant, I_irrelevant, I_conjunction
        elif args.index==3:
            # feature_conjunction_search
            I_single_feature = np.zeros((R,C,A))
            id = np.where(np.random.randint(0,2,(R,C)))
            I_single_feature[id[0],id[1],A//4] = 1
            I_single_feature[id[0],id[1],A//2] = 1
            I_single_feature[R//2, C//2, :] = 0
            I_single_feature[R//2, C//2, A//4] = 1
            I_single_feature[R//2, C//2, 0] = 1
            I_conjunction_feature = np.zeros((R,C,A))
            id = np.where(np.random.randint(0,2,(R,C)))
            I_conjunction_feature[id[0],id[1],A//4] = 1
            I_conjunction_feature[id[0],id[1],A//2] = 1
            id = np.where(np.random.randint(0,2,(R,C)))
            I_conjunction_feature[id[0],id[1],:] = 0
            I_conjunction_feature[id[0],id[1],A//4*3] = 1
            I_conjunction_feature[id[0],id[1],0] = 1
            I_conjunction_feature[R//2, C//2, :] = 0
            I_conjunction_feature[R//2, C//2, A//4] = 1
            I_conjunction_feature[R//2, C//2, 0] = 1
            return I_single_feature, I_conjunction_feature
        elif args.index==4:
            # border_segmentation
            I_pattern = np.zeros((R,C,A))
            I_pattern[:,:C//2,args.angle_1] = 1
            I_pattern[:,C//2:,args.angle_2] = 1
            return I_pattern
        elif args.index == 5:
            loc = np.random.randint(0,2,(R,C))
            I_pattern = np.zeros((R,C,A))
            I_pattern[np.where(loc)[0], np.where(loc)[1], np.random.randint(0,A,np.where(loc)[0].shape[0])] = 1
            for m in range(R):
                for n in range(C):
                    if round(np.sqrt((m-R//2)**2 + (n-C//2)**2))==3:
                        t = round(np.arctan2(n-C//2, m-R//2) * 180 / np.pi) // 15
                        I_pattern[m,n,:] = 0
                        I_pattern[m,n,t] = 1
            return I_pattern
        



def draw(I_pattern):
    
    ROI = (R//2, C//2, 9)
    # print(I_pattern)
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
    # beta, theta1, theta2 = V1.JW(5,5,0)
    # print(beta, theta1, theta2)
    X,Y = V1.run(1)
    # print(X[:,:,0])

    
    # print(np.max(X))
    # print(np.min(X))
    Pen_size = ((X-np.min(X))/(np.max(X)-np.min(X)))**1
    # print(Pen_size[:,:,0])
    Pen_size = (Pen_size - np.min(Pen_size))/(np.max(Pen_size)-np.min(Pen_size))*5
    # print(Pen_size[:,:,0])
    # 初始化海龟绘图窗口
    turtle.setup(width=C*20, height=R*20)
    turtle.speed(5000) # 设置最快绘制速度


    # draw a turtle image according to X
    def draw_turtle(X):
        for m in range(R):
            for n in range(C):
                for t in range(A):
                    if I_pattern[m][n][t] == 1:
                        # print(Pen_size[m][n][t])
                        turtle.pensize(Pen_size[m][n][t])
                        turtle.penup()
                        turtle.goto(m*20+10,n*20+10)
                        turtle.pendown()
                        turtle.setheading(t*15)
                        turtle.forward(5)
                        turtle.penup()
                        turtle.goto(m*20+10,n*20+10)
                        turtle.pendown()
                        turtle.setheading(t*15)
                        turtle.forward(-5)                    
                        turtle.penup()

    draw_turtle(X)
    turtle.title(f'zscore='+str(V1.zscore(ROI)))
    turtle.mainloop()
    # turtle.done()

def main(args):
    I_pattern = NeuronGroup.init_input_pattern(args)

    
    
    if args.index == 0:
        draw(I_pattern)
    elif args.index == 1:
        # draw(I_pattern[0])
        draw(I_pattern[1])
    elif args.index == 2:
        draw(I_pattern[0])
        draw(I_pattern[1])
        draw(I_pattern[2])
    elif args.index == 3:
        draw(I_pattern[0])
        draw(I_pattern[1])
    elif args.index >= 4:
        draw(I_pattern)
    


if __name__ == '__main__':
    np.random.seed(2023)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--index', '-i', type=int, default=0)
    argparser.add_argument('--center_angle', type=int, default=0)
    argparser.add_argument('--angle_1', type=int, default=A//4*3)
    argparser.add_argument('--angle_2', type=int, default=A//4)
    args = argparser.parse_args()

    # V1.check_W((R//2,C//2,0))
    main(args)


