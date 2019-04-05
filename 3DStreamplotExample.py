import numpy as np
import matplotlib
# If you want to rotate plot manually in a separate window, restart the console,
# comment out "%matplotlib inline" and use "matplotlib.use('TkAgg')"
matplotlib.use('TkAgg')
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from streamplot3D import streamplot3D

def plot_flow_field(ax, du_dt, dv_dt, dw_dt, u_range, v_range, w_range, p, density=10):
    """
    Plots the flow field with line thickness proportional to speed.
    """
    # Set up u,v space (which is u x v grids)
    u = np.linspace(u_range[0], u_range[1], density)
    v = np.linspace(v_range[0], v_range[1], density)
    w = np.linspace(w_range[0], w_range[1], density)
    uu, vv, ww = np.meshgrid(u, v, w)

    # Compute the vector based on ODE functions
    u_vel = np.zeros((len(u), len(v), len(w)))
    v_vel = np.zeros((len(u), len(v), len(w)))
    w_vel = np.zeros((len(u), len(v), len(w)))
    u_vel = du_dt(uu, vv, ww, p)
    v_vel = dv_dt(uu, vv, ww, p)
    w_vel = dw_dt(uu, vv, ww, p)

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2 + w_vel**2)

    # Make linewidths proportional to speed, with minimal line width of 0.5
    # and max of 5
    lw = 0.1 + 4.5 * speed / speed.max()
    x1 = [1,2,3]
    x2 = [2,3,6]
    # Make stream plot, here we need a function: streamplot3D
    # and the code here should be
    #start_points=np.array([x1,x2])
    ax = streamplot3D(ax, u, v, w, u_vel, v_vel, w_vel, linewidth=lw, arrowsize=1.5,
                        density=0.3, color=sns.color_palette()[0])

    return ax, u_vel, v_vel, w_vel

def dA_dt(A, B, C, p):
    """
    ODE of A
    """
    return p.alpha + p.beta * (2*A**2)**p.n / ((2*A**2)**p.n + (p.Kd + 4 * (A + B + C) + np.sqrt(p.Kd**2 + 8 * (A + B + C) * p.Kd))**p.n) - A

def dB_dt(A, B, C, p):
    """
    ODE of B
    """
    return p.alpha + p.beta * (2*B**2)**p.n / ((2*B**2)**p.n + (p.Kd + 4 * (A + B + C) + np.sqrt(p.Kd**2 + 8 * (A + B + C) * p.Kd))**p.n) - B

def dC_dt(A, B, C, p):
    """
    ODE of C
    """
    return p.alpha + p.beta * (2*C**2)**p.n / ((2*C**2)**p.n + (p.Kd + 4 * (A + B + C) + np.sqrt(p.Kd**2 + 8 * (A + B + C) * p.Kd))**p.n) - C

class BSParams(object):
    """
    Container for parameters for homodimeric bistable circuits.
    """
    def __init__(self, **kwargs):
        # Dimensionless parameters
        self.alpha = 0.1*1.5 # leakiness
        self.beta = 10*1.5 # Maximal production
        self.Kd = 1 # Dissociation constant of dimers
        self.m = 1 # A and B copy number ratio
        self.kappa = 1 # Ratio of A and B's binding affinity with DNA
        self.gamma = 1 # Ratio of stability between A and B
        self.n = 2 # Additional cooperativity of transcriptional activation

        # Put in params that were specified in input
        for entry in kwargs:
            setattr(self, entry, kwargs[entry])

# Set paramters
p = BSParams()
A_range = [p.alpha, 1.5*p.beta]
B_range = [p.alpha, 1.5*p.beta]
C_range = [p.alpha, 1.5*p.beta]

# Build figure
fig = plt.figure()
ax = Axes3D(fig)
ax, A_vel, B_vel, C_vel = plot_flow_field(ax, dA_dt, dB_dt, dC_dt, A_range, B_range, C_range, p, density=200)

# Tidy up
ax.set_xlim((0,A_range[1]))
ax.set_ylim((0,B_range[1]))
ax.set_zlim((0,C_range[1]))
ax.set_xlabel('$A$', fontsize=30)
ax.set_ylabel('$B$', fontsize=30, rotation=0);
ax.set_zlabel('$C$', fontsize=30, rotation=0);
ax.tick_params(labelsize=15)
ax.legend(['phase portrait'], fontsize=15,frameon=True,framealpha=1,loc='upper right',markerscale=5)
plt.show()

# Set paramters
# p = BSParams()
# fig = plt.figure()
# t = np.linspace(0, 100, 500)
# ax = fig.gca(projection='3d')
# A, B, C = np.meshgrid(np.linspace(0, p.beta, 5),
#                     np.linspace(0, p.beta, 5),
#                     np.linspace(0, p.beta, 5))
# u = dA_dt(A, B, C, p)
# v = dB_dt(A, B, C, p)
# w = dC_dt(A, B, C, p)
# ax.quiver(A, B, C, u, v, w, length=0.6, normalize=False)
# plt.show()
