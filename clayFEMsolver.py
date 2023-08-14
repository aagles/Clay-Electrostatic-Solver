#!/usr/bin/env python
# coding: utf-8

# # Attempting to Resolve the Electrostatic Forces on a single curved clay particle in solution

"""
The system will contain one large sheet with a user-inputed radius of curvature
It will compute the electrostatic potential in the surrounding solution
It will then compute the ionic concentration in solution given rho_infty
There will be no timestepping, at least for the moment
I should also then be able to visualize the force distribution on the clay surface elements
"""


import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy.constants as constants
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton
from mpl_toolkits.mplot3d import Axes3D
import os





# ## Initialize Background Mesh (Class), $\Omega$
# Create a class to represent the total mesh and to define the functions that need to be solved everywhere in the system
#

class Omega:
    def __init__(self, Lx, Ly, Lz, dx, dy, dz):
        self.Lx = Lx    # Size of the mesh in the x-direction
        self.Ly = Ly    # Size of the mesh in the y-direction
        self.Lz = Lz    # Size of the mesh in the z-direction
        self.dx = dx    # Dimension of the mesh elements in the x-direction
        self.dy = dy    # Dimension of the mesh elements in the y-direction
        self.dz = dz    # Dimension of the mesh elements in the z-direction


        # Calculate the number of elements in each direction
        self.Nx = int(self.Lx / self.dx)
        self.Ny = int(self.Ly / self.dy)
        self.Nz = int(self.Lz / self.dz)

        
        # Calculate the number of elements in each direction
        self.Nx = int(self.Lx / self.dx)
        self.Ny = int(self.Ly / self.dy)
        self.Nz = int(self.Lz / self.dz)
        
        # Initialize Variables for the SOLID mesh elements with numpy zeros
        self.solid = np.zeros((self.Nx, self.Ny, self.Nz)) # 1 where solid element, zero everywhere else
        self.surf = np.zeros((self.Nx, self.Ny, self.Nz))  # 1 where surf element, zero everywhere else
        self.sigma = np.zeros((self.Nx, self.Ny, self.Nz)) # sigma_value where solid, zero everywhere else
        self.psi = np.zeros((self.Nx, self.Ny, self.Nz))
        self.force = np.zeros((self.Nx, self.Ny, self.Nz))

        # Initialize Variables for the FLUID mesh elements with numpy zeros
        # self.rho_p = np.zeros((self.Nx, self.Ny, self.Nz))
        # self.rho_n = np.zeros((self.Nx, self.Ny, self.Nz))
        # self.mu = np.zeros((self.Nx, self.Ny, self.Nz))




    ### Properties Methods
    def get_dimensions(self):
        return self.Lx, self.Ly, self.Lz, self.dx, self.dy, self.dz

    def get_num_elements(self):
        return self.Nx, self.Ny, self.Nz

    
    ####################### INITIALIZATION ###############################################################

    ## SOLID Initialization Methods
    #### Initialize the indices for a solid element

    def initialize_solid(self, Sx, Sy, d, R, loc, oneD, sigma_value, read, dir):
        """
        Defining the self.solid and self.surf for a solid rectangular object with width Sx by Sy and thickness, d
        Centering the object at [x_pos, y_pos, z_pos].
        self.solid and self.surf values will be used later to define the boundary conditions for the PDE

        Sx, Sy, R, d, and loc are in units of the simulation cell (not in position in the mesh)
        x_pos, y_pos, and z_pos are the coordinates where the object should be centered.
        """
        if read:
            # read all the necessary files to skip the above mesh initialization
            self.solid = np.load(os.path.join(dir,'solid.npy'))
            self.surf  = np.load(os.path.join(dir,'surf.npy'))
            self.sigma = np.load(os.path.join(dir,'sigma.npy'))
        else:
            half_Sx = Sx / 2
            half_Sy = Sy / 2
            half_d  = d  / 2

            
            x = np.arange(0,self.Lx,self.dx)
            y = np.arange(0,self.Ly,self.dy)

            start_x = int(np.where(x > loc[0] - half_Sx)[0][0]) # find the first time x reaches the clay
            end_x = int(np.where(x > loc[0] + half_Sx)[0][0])

            if oneD: # if periodic in y:
                y = [0, self.dy]
                start_y = y[0]
                end_y = y[-1]
                half_Sy = self.dy / 2
            else:
                start_y = int(np.where(y > loc[1] - half_Sy)[0][0]) # find the first time y reaches the clay
                end_y = int(np.where(y > loc[1] + half_Sy)[0][0])

            for i in np.arange(start_x, end_x+1):
                for j in np.arange(start_y, end_y+1):
                    x_val = x[i] # x-coordinate
                    y_val = y[i] # y-coordinate

                    
                    z_squared = R**2 - (x_val - loc[0])**2   # - (y - y_pos)**2

                    if z_squared >= 0:
                        z = np.sqrt(z_squared) + loc[2] - R
                        start_index_z = int( (z - half_d) / self.dz )       
                        end_index_z = int( (z + half_d) / self.dz )

                        start_index_z = max(0, start_index_z)
                        end_index_z = min(self.Nz - 1, end_index_z)

                        # Label Solid elements
                        for kk in range(start_index_z, end_index_z + 1):
                            self.solid[i, j, kk] = 1

                        # Label Surface Elements
                        self.surf[i, j, start_index_z] = 1
                        self.surf[i, j, end_index_z] = 1

                        self.sigma[i, j, start_index_z] = sigma_value
                        self.sigma[i, j, end_index_z] = sigma_value

                        if ((int(i)==start_x or int(i)==end_x) or (int(j)==start_y or int(j)==end_y)):
                            for kkkk in range(start_index_z, end_index_z + 1):
                                self.surf[i, j, kkkk] = 1
                                self.sigma[i, j, kkkk] = 1
                            
                                    

    def save_mesh(self, dir):
        # save all the the necessary files as numpy arrays in the inputted dir

        np.save(dir+'solid.npy', self.solid)
        np.save(dir+'surf.npy', self.surf)
        np.save(dir+'sigma.npy', self.sigma)
   

   # Method to plot Psi
    def plot_solid2D(self, y_pos, oneD):
        # If 1-D case
        if oneD:
            y_pos = self.dy / 2
        # Check if y_index is within the domain
        if y_pos < 0 or y_pos >= self.Ly:
            print(f'Error: y_index should be in the range [0, {self.Ly}]', flush=True)
            return

        # Get the slice of the potential field at the given y-index
        y_index = int(round(y_pos/self.dy))
        psi_slice = self.solid[:, y_index, :]

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.imshow(psi_slice.T, origin='lower', extent=[0, self.Lx, 0, self.Lz], cmap='viridis')
        plt.colorbar(label='Potential V')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'Density map of potential $\Psi$ in xz plane at y-index {y_index}')
        plt.show()
    
    # Plotting the Object
    def plot_object(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        solid_indices = np.where(self.solid == 1)  # Get the indices where solid exists

        ax.scatter(solid_indices[0], solid_indices[1], solid_indices[2], c='b', marker='o')

        ax.set_xlim([0,self.Lx])
        ax.set_ylim([0,self.Ly])
        ax.set_zlim([0,self.Lz])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Representation of Solid Object')

        plt.show()
    



    ## FLUID Initialization Methods
    ###### Calculate the electrostatic potential, \Psi
    """
    Boundary Conditions:
                        Neumann at Omega edges
                        E_s = -sigma/eps/eps_0 at fluid solid interface
                        Psi value of all solid elements are set to psi_s to avoid discontinuities
                        This shouldn't impact the accuracy of the results as only the surface elements
                        are used to calculate the psi values of the fluid
    """
    def solve_fluid(self, rho0, T):
        """
        d^2(V)/dx^2 = (V[i-1] - 2V[i] + V[i-1]) / dx^2

        """

        # Calculate surface potential, Grahame Equation
        A = sigma_value**2 / (2*eps*constants.epsilon_0*constants.k*T*rho0)
        B = constants.e / (constants.k * T)
        psi_s = ( log(0.5*(A-sqrt(A+4)*sqrt(A)+2)) ) / B

        # Set initial conditions
        self.psi[self.surf == 1] = psi_s
        print(psi_s)

        # Define tolerance and maximum iterations for convergence
        tolerance = 1e-6
        max_iterations = 100

        for iteration in range(max_iterations):
            if (iteration % 1) == 0:
                print('Iteration '+str(iteration), flush=True)
                np.save('current_psi.npy', self.psi)


            # Create a copy of the potential field to calculate the change
            psi_old = self.psi.copy()

            # Update the potential field
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    for k in range(1, self.Nz - 1):
                        if self.solid[i, j, k] == 0: # Only update for fluid points

                            RHS = (((rho0*constants.e)/(eps*constants.epsilon_0)) 
                                 * (np.exp(-constants.e*psi_old[i,j,k]/(constants.k*T)) - np.exp(constants.e*psi_old[i,j,k]/(constants.k*T))))
                            
                            # for eqn with the form: (2x+A)/a^2 + (2x+B)/b^2 + (2x+C)/c^2 = D
                            # x = [Da^2b^2c^2 - Ab^2c^2 - Ba^2c^2 - Ca^2b^2] / 2(b^2c^2 + a^2c^2 + a^2b^2)
                            A = psi_old[i+1,j,k] + psi_old[i-1,j,k]
                            a = self.dx
                            B = psi_old[i,j+1,k] + psi_old[i,j-1,k]
                            b = self.dy
                            C = psi_old[i,j,k+1] + psi_old[i,j,k-1]
                            c = self.dz
                            D = RHS 
                            numerator = (D * a**2 * b**2 * c**2) - (A * b**2 * c**2 + B * a**2 * c**2 + C * a**2 * b**2)
                            denominator = 2 * (b**2 * c**2 + a**2 * c**2 + a**2 * b**2)

                            self.psi[i, j, k] = numerator / denominator

            # Apply Neumann boundary conditions at the edges of the domain
            self.psi[0, :, :] = self.psi[1, :, :]
            self.psi[-1, :, :] = self.psi[-2, :, :]
            self.psi[:, 0, :] = self.psi[:, 1, :]
            self.psi[:, -1, :] = self.psi[:, -2, :]
            self.psi[:, :, 0] = self.psi[:, :, 1]
            self.psi[:, :, -1] = self.psi[:, :, -2]

            # Check for convergence
            if np.max(np.abs(psi_old - self.psi)) < tolerance:
                print(f'Converged after {iteration} iterations')
                break

        if iteration == max_iterations - 1:
            print('Warning: solve_laplace did not converge')


    # Method to save the Psi map
    def save_psi(self, fname):
        np.save(fname, self.psi)


# Method to plot Psi
    def plot_psi(self, y_pos):
        # Check if y_index is within the domain
        if y_pos < 0 or y_pos >= self.Ly:
            print(f'Error: y_index should be in the range [0, {self.Ly}]', flush=True)
            return

        # Get the slice of the potential field at the given y-index
        y_index = int(round(y_pos/self.dy))
        psi_slice = self.psi[:, y_index, :]

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.imshow(psi_slice.T, origin='lower', extent=[0, self.Lx, 0, self.Lz], cmap='viridis')
        plt.colorbar(label='Potential V')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'Density map of potential $\Psi$ in xz plane at y-index {y_index}')
        # plt.show()
        


# Some Constants:
eps         = 80   # dielectric constant of water
sigma_value = -0.1 # charge density of a single mesh element (C/m^3)
rho0        = 0.01 # number density of ions in the system
T           = 300  # temperature (K)

# Steps to implement:
oneD=False
omega = Omega(Lx=20, Ly=20, Lz=20, dx=.1, dy=.1, dz=.1)
omega.initialize_solid(Sx=10, Sy=10, d=.2, R=8, loc=[10,10,10], oneD=oneD, sigma_value=sigma_value, read=False, dir='../mesh/')
omega.save_mesh('../mesh/')
# omega.plot_solid2D(y_pos=10, oneD=oneD)
# omega.plot_object()
# omega.solve_fluid(rho0=rho0, T=T)
# omega.save_psi('psi_map.npy')
# omega.plot_psi(y_pos=25)






