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
from multiprocessing import Pool, freeze_support
import os 
from numba import jit, prange



# Function to solve subdomains
def solve_subdomain(args):
    subdomain, psi_old, solid, rho0, T, a, b, c = args

    psi_domain = psi_old.copy() #[subdomain[0]:subdomain[1],:,:]
    # Update the potential field
    x_start = subdomain[0]+1 if subdomain[0]==0 else subdomain[0]  # want the second index for first domain
    x_end = subdomain[1]-1 if subdomain[1]==psi_old.shape[0] else subdomain[1] # want to take the penultimate index for last subdomain

    for i in range(x_start, x_end):
        for j in range(1, subdomain[2] - 1):
            for k in range(1, subdomain[3] - 1):
                if solid[i, j, k] == 0: # Only update for fluid points

                    RHS = (((rho0*constants.e)/(eps*constants.epsilon_0)) 
                            * (np.exp(-constants.e*psi_old[i,j,k]/(constants.k*T)) - np.exp(constants.e*psi_old[i,j,k]/(constants.k*T))))
                    
                    # Solving for Psi[i,j,k]
                    # for eqn with the form: (2x+A)/a^2 + (2x+B)/b^2 + (2x+C)/c^2 = D
                    # x = [Da^2b^2c^2 - Ab^2c^2 - Ba^2c^2 - Ca^2b^2] / 2(b^2c^2 + a^2c^2 + a^2b^2)
                    A = psi_old[i+1,j,k] + psi_old[i-1,j,k]
                    B = psi_old[i,j+1,k] + psi_old[i,j-1,k]
                    C = psi_old[i,j,k+1] + psi_old[i,j,k-1]
                    D = RHS 
                    numerator = (D * a**2 * b**2 * c**2) - (A * b**2 * c**2 + B * a**2 * c**2 + C * a**2 * b**2)
                    denominator = 2 * (b**2 * c**2 + a**2 * c**2 + a**2 * b**2)

                    psi_domain[i, j, k] = numerator / denominator

    # Apply Neumann boundary conditions at the edges of the domain
    # psi_new[0, :, :] = psi_new[1, :, :]
    # psi_new[-1, :, :] = psi_new[-2, :, :]
    # psi_new[:, 0, :] = psi_new[:, 1, :]
    # psi_new[:, -1, :] = psi_new[:, -2, :]
    # psi_new[:, :, 0] = psi_new[:, :, 1]
    # psi_new[:, :, -1] = psi_new[:, :, -2]

    return psi_domain[x_start:x_end,:,:]


# ## Initialize Background Mesh (Class), $\Omega$
# Create a class to represent the total mesh and to define the functions that need to be solved everywhere in the system
#
class Omega:
    def __init__(self, Lx, Ly, Lz, dx, dy, dz):
        # size of the mesh in units of nm
        self.Lx = Lx    # Size of the mesh in the x-direction
        self.Ly = Ly    # Size of the mesh in the y-direction
        self.Lz = Lz    # Size of the mesh in the z-direction
        self.dx = dx    # Dimension of the mesh elements in the x-direction, also in nm
        self.dy = dy    # Dimension of the mesh elements in the y-direction
        self.dz = dz    # Dimension of the mesh elements in the z-direction


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
            self.sigma = self.surf * sigma_value
        else:
            half_Sx = Sx / 2
            half_Sy = Sy / 2
            half_d  = d  / 2

            for i in range(self.Nx):
                for j in range(self.Ny):
                    for k in range(self.Nz):
                        x = i * self.dx # x-coordinate
                        y = j * self.dy # y-coordinate
                        z = k * self.dz # z-coordinate

                        if abs(x - loc[0]) <= half_Sx and abs(y - loc[1]) <= half_Sy:
                            z_squared = R**2 - (x - loc[0])**2   # - (y - y_pos)**2

                            if z_squared >= 0:
                                z = int(np.sqrt(z_squared) + loc[2] - R)
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

    def save_mesh(self, dir):
        # save all the the necessary files as numpy arrays in the inputted dir

        np.save(dir+'solid.npy', self.solid)
        np.save(dir+'surf.npy', self.surf)
    

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

    
        
        

    def solve_fluid_parallel(self, rho0, T, num_processes):
        """
        d^2(V)/dx^2 = (V[i+1] - 2V[i] + V[i-1]) / dx^2

        """
        # Calculate surface potential, Grahame Equation
        # A = sigma_value**2 / (2*eps*constants.epsilon_0*constants.k*T*rho0)
        B = constants.e / (constants.k * T)
        # psi_s = ( log(0.5*(A+sqrt(A+4)*sqrt(A)+2)) ) / B
        C = sigma_value**2 / (4*constants.k*T*constants.epsilon_0*eps*rho0)
        psi_s = ( acosh(C + 1) ) / B

        # Set initial conditions
        self.psi[self.surf == 1] = psi_s
        print(psi_s)

        # Subdomain Information
        subdomains = [(i*self.Nx//num_processes, (i+1)*self.Nx//num_processes, self.Ny, self.Nz) for i in range(num_processes)]

        # Define tolerance and maximum iterations for convergence
        max_iterations = 5000
        abs_error = np.empty(max_iterations)

        for iteration in range(max_iterations):
            if (iteration % 10) == 0:
                print('Iteration '+str(iteration))
                np.save('../data/current_psi_'+name+'.npy', self.psi)
            elif (iteration == max_iterations-1):
                np.save('../data/psi_map_'+name+'.npy', self.psi)

            # Create a copy of the potential field to calculate the change
            psi_old = self.psi.copy()
            
            # call the parallel calculation of the entire domain
            #args = (lower_x, upper_x, Ny, Nz, psi_old, solid, rho0, T, a, b, c)
            a = self.dx * (1e-9) # converting to units of m
            b = self.dy * (1e-9)
            c = self.dz * (1e-9)
            args = [(subdomain, psi_old, self.solid, rho0, T, a, b, c) for subdomain in subdomains]
            
            with Pool(num_processes) as p:
                results = p.map(solve_subdomain, args)

            # combine results from all subdomains
            for i, result in enumerate(results):
                
                if subdomains[i][0]==0: # if first subdomain
                    x_start = subdomains[i][0]+1
                    x_end = subdomains[i][1]
                    self.psi[x_start:x_end, :, :]  = result
                elif subdomains[i][1]==psi_old.shape[0]:# if the last subdomain
                    x_start = subdomains[i][0]
                    x_end = subdomains[i][1]-1  
                    self.psi[x_start:x_end, :, :] = result
                else: 
                    x_start = subdomains[i][0]
                    x_end = subdomains[i][1]
                    self.psi[x_start:x_end, :, :] = result

            
                
                # Apply Neumann boundary conditions at the edges of the domain
                self.psi[0, :, :] = self.psi[1, :, :]
                self.psi[-1, :, :] = self.psi[-2, :, :]
                self.psi[:, 0, :] = self.psi[:, 1, :]
                self.psi[:, -1, :] = self.psi[:, -2, :]
                self.psi[:, :, 0] = self.psi[:, :, 1]
                self.psi[:, :, -1] = self.psi[:, :, -2]


            # Record rate of convergence
            abs_error[iteration] = np.max(np.abs(psi_old - self.psi))

            # Check for convergence
            if abs_error[iteration] > abs_error[iteration-1]:
                print(f'Converged after {iteration} iterations')
                break
            


        if iteration == max_iterations - 1:
            print('Warning: solve_laplace did not converge')

        np.save('../data/abs_error_'+name+'.npy', abs_error)


    # Method to save the Psi map
    def save_psi(self, fname):
        np.save(fname, self.psi)

    # Method to plot Psi
    def plot_psi(self, y_pos, read, fn):
        # Check if y_index is within the domain
        if y_pos < 0 or y_pos >= self.Ly:
            print(f'Error: y_index should be in the range [0, {self.Ly}]')
            return

        # Get the slice of the potential field at the given y-index
        y_index = int(round(y_pos/self.dy))
        if read:
            psi_slice = np.load(fn)[:, y_index, :]
        else:
            psi_slice = self.psi[:, y_index, :]
        
        psi_slice = psi_slice * 1000 # converting to mV

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.imshow(psi_slice.T, origin='lower', extent=[0, self.Lx, 0, self.Lz], cmap='viridis') #, vmin=-0.2, vmax=0.8)
        plt.colorbar(label='Potential $\Psi (mV)$')
        plt.xlabel('x (nm)')
        plt.ylabel('z (nm)')
        plt.title(f'Density map of potential $\Psi (mV)$ in xz plane at y-index {y_index}')
        plt.savefig('../data/psi_final.png', dpi=300)
        plt.show()
        



       




# Some Constants:
eps         = 80   # dielectric constant of water
sigma_value = -0.2 # charge density of a single mesh element (C/m^3)
rho0_M      = 1  # inputted rho density of ions in the system (mol/L)
rho0        = rho0_M * constants.Avogadro * 1000  # rho density in units of ions/m^3
T           = 300  # temperature (K)
name        = 'rho0_1'

# Steps to implement:
oneD=False

if __name__ == '__main__':
    freeze_support()
    omega = Omega(Lx=20, Ly=20, Lz=20, dx=.1, dy=.1, dz=.1)
    omega.initialize_solid(Sx=10, Sy=10, d=.2, R=8, loc=[10,10,10], oneD=oneD, sigma_value=sigma_value, read=True, dir='../mesh/')
    # omega.save_mesh('/Volumes/GoogleDrive/My Drive/research/projects/LBNL/ClayPBEsolver/mesh')
    # omega.plot_object()
    omega.solve_fluid_parallel(rho0=rho0, T=T, num_processes=8)
    omega.save_psi('../data/psi_map_'+name+'.npy')
    # omega.plot_psi(y_pos=10, read=True, fn='../data/final_psi_rho0_'+str(rho0_M)+'.npy')






