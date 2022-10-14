from turtle import width
import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        output_size_x = self.A.shape[2] - self.kernel + 1
        output_size_y = self.A.shape[3] - self.kernel + 1
        

        Z = np.zeros((self.A.shape[0], self.A.shape[1], output_size_x, output_size_y))
        self.max__idx_i = np.zeros(Z.shape)
        self.max__idx_j = np.zeros(Z.shape)
        
        
        for i in range(output_size_x):
            end_i = self.kernel+i
            for j in range(output_size_y):
                end_j = j+self.kernel
                block_A = self.A[:,:,i:end_i, j:end_j]
                Z[:,:,i,j] = np.max(block_A, axis=(-1,-2))
                block_A_values = block_A.reshape(*block_A.shape[:-2], -1)
                argmax_i =  np.argmax(block_A_values, axis=-1)
                self.max__idx_i[:,:,i,j] = argmax_i//self.kernel + i
                argmax_j = np.argmax(block_A_values, axis=-1)       
                self.max__idx_j[:,:,i,j] = argmax_j%self.kernel + j

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
      

        dLdA = np.zeros((self.A.shape[0], self.A.shape[1], self.A.shape[2], self.A.shape[3]))

        for i in range(dLdZ.shape[2]): # i
            for j in range(dLdZ.shape[3]): # j
                for x in range(dLdZ.shape[0]): # batch
                    for y in range(dLdZ.shape[1]): # channel
                        idx_i = int(self.max__idx_i[x, y, i, j])
                        idx_j = int(self.max__idx_j[x, y, i, j])
                        dLdA[x, y, idx_i, idx_j] += dLdZ[x, y, i, j]
        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        batch_size, in_channels, input_size_x, input_size_y = self.A.shape

        output_x = input_size_x - self.kernel + 1
        output_y = input_size_y - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_x, output_y))
        filter_size = self.kernel*self.kernel

        for i in range(output_x):
            step_i = i+self.kernel
            for j in range(output_y):
                step_j = j+self.kernel
                Z[:,:,i,j] = np.sum(self.A[:,:,i:step_i, j:step_j], axis=(-1,-2))/filter_size

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdA = np.zeros((self.A.shape[0], self.A.shape[1], self.A.shape[2], self.A.shape[3]))

        kernel_size = self.kernel*self.kernel

        for i in range(self.kernel):
            for j in range(self.kernel):
                dLdA[:,:,i:i+dLdZ.shape[2],j:j+dLdZ.shape[3]] = dLdA[:,:,i:i+dLdZ.shape[2],j:j+dLdZ.shape[3]]+dLdZ/kernel_size

        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel) #TODO
        self.downsample2d = Downsample2d(self.stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
            
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel) #TODO
        self.downsample2d = Downsample2d(self.stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)
        return dLdA
