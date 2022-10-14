import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor


    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        
        batch_size, in_channel, input_size = A.shape
        output_size = A.shape[2]* self.upsampling_factor-( self.upsampling_factor-1) # TODO
        print(A.shape, output_size)
        self.A = A
        Z = np.zeros((batch_size, in_channel, output_size))
        Z[:, :,: : self.upsampling_factor] = A
 
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # batch_size, in_channel, input_size = dLdZ.shape
        # output_size = (dLdZ.shape[2]+(self.upsampling_factor-1))/self.upsampling_factor
        dLdA = dLdZ[:, :,: : self.upsampling_factor]
        
        

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        Z = A[:, :,: : self.downsampling_factor]
        self.A = A
        
        print('input shape',A.shape)

        # Z = None # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channel, input_size = dLdZ.shape
       
        print('dldz shape', dLdZ.shape)
        print('A shape', self.A.shape)
        # if input_size%2!=0:
        output_size = dLdZ.shape[2]* self.downsampling_factor-( self.downsampling_factor-1) # TODO
        

        dLdA = np.zeros((batch_size, in_channel, output_size))
        dLdA[:, :,: : self.downsampling_factor] = dLdZ
        if dLdA.shape[2]!=self.A.shape[2]:
            print('hey')
            diff = abs(dLdA.shape[2]-self.A.shape[2])
            print(diff)
            dLdA = np.pad(dLdA, ((0, 0), (0, 0), (0 , diff)), 'constant', 
                 constant_values=0)
        # import pdb 



        # ((0, 0), )
        # pdb.set_trace()
        print('dlda shape',dLdA.shape)
        # dLdA = None  #TODO

        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        print(A.shape)
        batch_size, in_channel, input_size_x, input_size_y = A.shape
        output_size_x = input_size_x* self.upsampling_factor-( self.upsampling_factor-1) # TODO
        output_size_y = input_size_y* self.upsampling_factor-( self.upsampling_factor-1) # TODO

        # print(output_size_x, output_size_y)

        Z = np.zeros((batch_size, in_channel, output_size_x, output_size_y))

        
        Z[:, :,: : self.upsampling_factor, : : self.upsampling_factor] = A
        # print(res)


        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        # Z = None # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = dLdZ[:, :,: : self.upsampling_factor, : : self.upsampling_factor]

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        print('A shape', A.shape)
        self.A = A
        Z = A[:, :,: : self.downsampling_factor, : : self.downsampling_factor]
        print('Z shape',Z.shape)

        # Z = None # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, in_channel, input_size_x, input_size_y = dLdZ.shape
        print('dldz shape', dLdZ.shape)
        
        # if input_size_x%2!=0:
        output_size_x = input_size_x* self.downsampling_factor-( self.downsampling_factor-1) # TODO
        
        output_size_y = input_size_y* self.downsampling_factor-( self.downsampling_factor-1) # TODO


        dLdA = np.zeros((batch_size, in_channel, output_size_x, output_size_y))
        print('DLDA shape',dLdA.shape)
        dLdA[:, :,: : self.downsampling_factor, : : self.downsampling_factor] = dLdZ
        if dLdA.shape[2]!=self.A.shape[2]:
            print('hey')
            diff = abs(dLdA.shape[2]-self.A.shape[2])
            print(diff)
            dLdA = np.pad(dLdA, ((0, 0), (0, 0), (0 , diff), (0 , diff)), 'constant', 
                 constant_values=0)
        
        # print('DLDA shape',dLdA)
        # dLdA = None  #TODO

        return dLdA
