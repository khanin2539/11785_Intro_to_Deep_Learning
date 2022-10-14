import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)


    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
 
        
        # # Add your code here.
        # # Define your variables based on the writeup using the corresponding
        # # names below.
        
        # # This code should not take more than 10 lines. 
        # self.r = self.r_act(np.dot(self.Wrx, self.x)+self.brx+np.dot(self.Wrh, h)+self.brh)
        # self.z = self.z_act(np.dot(self.Wzx, self.x)+self.bzx+np.dot(self.Wzh, h)+self.bzh)
        # a = np.dot(self.Wnx, self.x)
        # b = self.r*(np.dot(self.Wnh, h)+self.bnh)
        # self.n = self.h_act(a+self.bnx+b)
        # h_t = ((1-self.z)*self.n)+(self.z*h)

       
        self.x = x
        self.hidden = h

        x = np.expand_dims(x, 1) 
        h = np.expand_dims(h, 1)

        self.brx = self.brx.reshape(-1, 1)
        self.brh = self.brh.reshape(-1, 1)
        self.bzh = self.bzh.reshape(-1, 1)
        self.bnx = self.bnx.reshape(-1, 1)
        self.bnh = self.bnh.reshape(-1, 1)
        

        self.r = self.r_act(
                np.dot(self.Wrx, x )+ self.brx + np.dot(self.Wrh,  h )+ self.brh)
        self.z = self.z_act(
                np.dot(self.Wzx,  x) + self.bzx.reshape((-1, 1)) + np.dot(self.Wzh,  h )+ self.bzh)
        self.n = self.h_act(np.dot(self.Wnx, x) + self.bnx.reshape((-1, 1)) + self.r * (
                np.dot(self.Wnh,  h) +self.bnh))
        h_t = (1 - self.z) * self.n + self.z * h

        self.r = np.squeeze(self.r)
        print(self.r)
        self.z = np.squeeze(self.z)
        print(self.z)
        self.n = np.squeeze(self.n)
        print(self.n)
        h_t = np.squeeze(h_t)

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)

        return h_t
        
        

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        
        delta = np.reshape(delta, (-1, 1))
       

        r = np.reshape(self.r, (-1, 1))
        z = np.reshape(self.z, (-1, 1))
        n = np.reshape(self.n, (-1, 1))
        h_prev = np.reshape(self.hidden, (-1, 1))
        x = np.reshape(self.x, (-1, 1)).transpose()  
        h = np.expand_dims(self.hidden, 1)

        print(x.shape)
        print(self.hidden.shape)

        
        dn = delta * (1 - z)
        delta_d_n = dn * self.h_act.derivative(n)
        dz = delta * (-n + h_prev)
        dr = delta_d_n * (np.dot(self.Wnh, h) + self.bnh.reshape(-1, 1))

        
        self.dWnx = np.dot(delta_d_n , x)
        self.dbnx = np.squeeze(delta_d_n)
        
        self.dWnh = np.dot(delta_d_n * r,  h_prev.transpose())
        self.dbnh = np.squeeze(delta_d_n * r)
        

        delta_dr = dr * self.r_act.derivative()
        self.dWrx = np.dot(delta_dr,  x)
        self.dbrx = np.squeeze(delta_dr)
        self.dWrh = np.dot(delta_dr , h_prev.transpose())
        self.dbrh = np.squeeze(delta_dr)

        delta_dz = dz * self.z_act.derivative()
        self.dWzx = np.dot(delta_dz , x)
        self.dbzx = np.squeeze(delta_dz)
        self.dWzh = np.dot(delta_dz,  h_prev.transpose())
        self.dbzh = np.squeeze(delta_dz)

        

        dx = np.zeros((1, self.d))
        dx += np.dot(delta_d_n.transpose(),  self.Wnx)
        dx += np.dot(delta_dz.transpose() , self.Wzx)
        dx += np.dot(delta_dr.transpose() , self.Wrx)

        dh_prev = np.zeros((1, self.h))
        dh_prev += (delta * z).transpose()
        dh_prev += np.dot((delta_d_n * r).transpose() , self.Wnh)
        dh_prev += np.dot(delta_dz.transpose() , self.Wzh)
        dh_prev += np.dot(delta_dr.transpose() ,self.Wrh)

        return dx, dh_prev
        



