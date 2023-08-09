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



# Function to solve subdomains
def solve_subdomain(args):
    subdomain, psi_old, solid, rho0, T, a, c = args

    psi_domain = psi_old.copy() #[subdomain[0]:subdomain[1],:,:]
    # Update the potential field
    x_start = subdomain[0]+1 if subdomain[0]==0 else subdomain[0]  # want the second index for first domain
    x_end = subdomain[1]-1 if subdomain[1]==psi_old.shape[0] else subdomain[1] # want to take the penultimate index for last subdomain

    for i in range(x_start, x_end):
        for k in range(1, subdomain[2] - 1):
            if solid[i, k] == 0: # Only update for fluid points

                RHS = (((rho0*constants.e)/(eps*constants.epsilon_0)) 
                        * (np.exp(-constants.e*psi_old[i,k]/(constants.k*T)) - np.exp(constants.e*psi_old[i,k]/(constants.k*T))))
                
                # Solving for Psi[i,j,k]
                # for eqn with the form: (2x+A)/a^2 + (2x+C)/c^2 = D
                # x = [Da^2b^2c^2 - Ac^2 - Ca^2] / 2(c^2 + a^2c^2 + a^2)
                A = psi_old[i+1,k] + psi_old[i-1,k]
                C = psi_old[i,k+1] + psi_old[i,k-1]
                D = RHS 
                numerator = (D * a**2 * c**2) - (A * c**2 +  C * a**2)
                denominator = 2 * (c**2 + a**2 * c**2 + a**2)

                psi_domain[i, k] = numerator / denominator


    return psi_domain[x_start:x_end,:]


# ## Initialize Background Mesh (Class), $\Omega$
# Create a class to represent the total mesh and to define the functions that need to be solved everywhere in the system
#
class Omega:
    def __init__(self, Lx, Lz, dx, dz):
        # size of the mesh in units of nm
        self.Lx = Lx    # Size of the mesh in the x-direction
        self.Lz = Lz    # Size of the mesh in the z-direction
        self.dx = dx    # Dimension of the mesh elements in the x-direction, also in nm
        # self.dy = dy    # Dimension of the mesh elements in the y-direction
        self.dz = dz    # Dimension of the mesh elements in the z-direction


        # Calculate the number of elements in each direction
        self.Nx = int(self.Lx / self.dx)
        self.Nz = int(self.Lz / self.dz)

        
        # Initialize Variables for the SOLID mesh elements with numpy zeros

        self.solid = np.zeros((self.Nx, self.Nz)) # 1 where solid element, zero everywhere else
        self.surf = np.zeros((self.Nx, self.Nz))  # 1 where surf element, zero everywhere else
        self.sigma = np.zeros((self.Nx, self.Nz)) # sigma_value where solid, zero everywhere else
        self.psi = np.zeros((self.Nx, self.Nz))
        self.force = np.zeros((self.Nx, self.Nz))

        # Initialize Variables for the FLUID mesh elements with numpy zeros
        # self.rho_p = np.zeros((self.Nx, self.Ny, self.Nz))
        # self.rho_n = np.zeros((self.Nx, self.Ny, self.Nz))
        # self.mu = np.zeros((self.Nx, self.Ny, self.Nz))




    ### Properties Methods
    def get_dimensions(self):
        return self.Lx, self.Lz, self.dx, self.dy, self.dz

    def get_num_elements(self):
        return self.Nx, self.Nz

    
    ####################### INITIALIZATION ###############################################################

    ## SOLID Initialization Methods
    #### Initialize the indices for a solid element

    def initialize_solid(self, Sx, d, R, loc, sigma_value, read, dir):
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
            half_d  = d  / 2

            x = np.arange(0,self.Lx,self.dx)

            start_x = int(np.where(x > loc[0] - half_Sx)[0][0]) # find the first time x reaches the clay
            end_x = int(np.where(x > loc[0] + half_Sx)[0][0])


            for i in np.arange(start_x, end_x+1):
                x_val = x[i] # x-coordinate

                
                z_squared = R**2 - (x_val - loc[0])**2   # - (y - y_pos)**2

                if z_squared >= 0:
                    z = np.sqrt(z_squared) + loc[1] - R
                    start_index_z = int( (z - half_d) / self.dz )       
                    end_index_z = int( (z + half_d) / self.dz )

                    start_index_z = max(0, start_index_z)
                    end_index_z = min(self.Nz - 1, end_index_z)

                    # Label Solid elements
                    for kk in range(start_index_z, end_index_z + 1):
                        self.solid[i, kk] = 1

                    # Label Surface Elements
                    self.surf[i, start_index_z] = 1
                    self.surf[i, end_index_z] = 1

                    self.sigma[i, start_index_z] = sigma_value
                    self.sigma[i, end_index_z] = sigma_value

                    if (int(i)==start_x or int(i)==end_x):
                        for kkkk in range(start_index_z, end_index_z + 1):
                            self.surf[i, kkkk] = 1
                            self.sigma[i, kkkk] = 1

    def save_mesh(self, dir):
        # save all the the necessary files as numpy arrays in the inputted dir
        # if ~os.path.exists(dir): os.system('mkdir '+dir)

        np.save(dir+'solid.npy', self.solid)
        np.save(dir+'surf.npy', self.surf)
    

    # Plotting the Object
    def plot_object(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        plt.imshow(self.solid.T, origin='lower', extent=[0, self.Lx, 0, self.Lz], cmap='viridis') #, vmin=-0.2, vmax=0.8)
        plt.colorbar(label='Potential $\Psi (mV)$')
        plt.xlabel('x (nm)')
        plt.ylabel('z (nm)')
        plt.title(f'Shape of 2D object in xz plane')
        plt.savefig('../data/2d_solid.png', dpi=300)
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
        subdomains = [(i*self.Nx//num_processes, (i+1)*self.Nx//num_processes, self.Nz) for i in range(num_processes)]

        # Define tolerance and maximum iterations for convergence
        tolerance = 1e-6
        max_iterations = 1000
        abs_error = np.empty(max_iterations)

        for iteration in range(max_iterations):
            if (iteration % 10) == 0:
                print('Iteration '+str(iteration))
                np.save('../data/2D_current_psi'+name+'.npy', self.psi)
            elif (iteration == max_iterations-1):
                np.save('../data/2D_psi_map'+name+'.npy', self.psi)

            # Create a copy of the potential field to calculate the change
            psi_old = self.psi.copy()
            
            # call the parallel calculation of the entire domain
            #args = (lower_x, upper_x, Nz, psi_old, solid, rho0, T, a, b, c)
            a = self.dx * (1e-9) # converting to units of m
            c = self.dz * (1e-9)
            args = [(subdomain, psi_old, self.solid, rho0, T, a, c) for subdomain in subdomains]
            
            with Pool(num_processes) as p:
                results = p.map(solve_subdomain, args)

            # combine results from all subdomains
            for i, result in enumerate(results):
                
                if subdomains[i][0]==0: # if first subdomain
                    x_start = subdomains[i][0]+1
                    x_end = subdomains[i][1]
                    self.psi[x_start:x_end, :]  = result
                elif subdomains[i][1]==psi_old.shape[0]:# if the last subdomain
                    x_start = subdomains[i][0]
                    x_end = subdomains[i][1]-1  
                    self.psi[x_start:x_end, :] = result
                else: 
                    x_start = subdomains[i][0]
                    x_end = subdomains[i][1]
                    self.psi[x_start:x_end, :] = result

            
                
                # Apply Neumann boundary conditions at the edges of the domain
                self.psi[0, :] = self.psi[1, :]
                self.psi[-1, :] = self.psi[-2, :]
                self.psi[:, 0] = self.psi[:, 1]
                self.psi[:, -1] = self.psi[:, -2]

            # Check for convergence
            # if np.max(np.abs(psi_old - self.psi)) < tolerance:
            #     print(f'Converged after {iteration} iterations')
            #     break

            # Record rate of convergence
            abs_error[iteration] = np.max(np.abs(psi_old - self.psi))
            


        if iteration == max_iterations - 1:
            print('Warning: solve_laplace did not converge')

        np.save('../data/2D_abs_error_'+name+'.npy', abs_error)


    # Method to save the Psi map
    def save_psi(self, fname):
        np.save(fname, self.psi)

    # Method to plot Psi
    def plot_psi(self, read, fn):
        # Get the slice of the potential field at the given y-index
        if read:
            psi_slice = np.load(fn)[:, :]
        else:
            psi_slice = self.psi[:, :]
        
        psi_slice = psi_slice * 1000 # converting to mV

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.imshow(psi_slice.T, origin='lower', extent=[0, self.Lx, 0, self.Lz], cmap='viridis') #, vmin=-0.2, vmax=0.8)
        plt.colorbar(label='Potential $\Psi (mV)$')
        plt.xlabel('x (nm)')
        plt.ylabel('z (nm)')
        plt.title(f'Density map of potential $\Psi (mV)$ in xz plane')
        plt.savefig('../data/2D_psi_final.png', dpi=300)
        plt.show()
        



       




# Some Constants:
eps         = 80   # dielectric constant of water
sigma_value = -6.03E-3 # charge density of a single mesh element (C/m^3)
rho0_M      = 1  # inputted rho density of ions in the system (mol/L)
rho0        = rho0_M * constants.Avogadro * 1000  # rho density in units of ions/m^3
T           = 300  # temperature (K)
name        = 'rho0_1'



if __name__ == '__main__':
    freeze_support()
    omega = Omega(Lx=20, Lz=20, dx=.01, dz=.01)
    omega.initialize_solid(Sx=10, d=.2, R=8, loc=[10,10], sigma_value=sigma_value, read=True, dir='../2Dmesh_hr/')
    # omega.save_mesh('/Volumes/GoogleDrive/My Drive/research/projects/LBNL/ClayPBEsolver/2Dmesh_hr/')
    # omega.plot_object()
    omega.solve_fluid_parallel(rho0=rho0, T=T, num_processes=4)
    # omega.save_psi('../data/2D_psi_map_'+name+'.npy')
    # omega.plot_psi(y_pos=10, read=True, fn='../data/final_psi_rho0_'+str(rho0_M)+'.npy')






