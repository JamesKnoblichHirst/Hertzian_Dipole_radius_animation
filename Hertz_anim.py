import numpy as np
from scipy import constants
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def update(frame_idx, E, r, X, Y, Z, wavelength, ax, surf):
    ax.clear()
    ax.set_xlim([-10*wavelength,10*wavelength])
    ax.set_ylim([-10*wavelength,10*wavelength])
    ax.set_zlim([-10*wavelength,10*wavelength])
    ax.view_init(elev=30, azim=30)
    ax.set_axis_off()

    E_slice = E[frame_idx, :, :, :]  # (theta, phi, xyz)

    # Faster magnitude calculation with einsum
    E_mag = np.einsum('ijk,ijk->ij', E_slice, E_slice)**0.5

    E_norm = np.abs(E_mag) / np.max(np.abs(E_mag))

    r_now = r[frame_idx]
    X_plot = r_now * X * E_norm
    Y_plot = r_now * Y * E_norm
    Z_plot = r_now * Z * E_norm

    surf[0] = ax.plot_surface(X_plot, Y_plot, Z_plot,
                              facecolors=plt.cm.viridis(E_norm),
                              rstride=1, cstride=1,
                              linewidth=0, antialiased=True)

    return surf



def spherical_unit_vectors(THETA, PHI):
    sin_theta = np.sin(THETA)
    cos_theta = np.cos(THETA)
    sin_phi = np.sin(PHI)
    cos_phi = np.cos(PHI)

    r_hat = np.stack([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    ], axis=-1)  # (N, M, 3)

    theta_hat = np.stack([
        cos_theta * cos_phi,
        cos_theta * sin_phi,
        -sin_theta
    ], axis=-1)

    phi_hat = np.stack([
        -sin_phi,
        cos_phi,
        np.zeros_like(PHI)
    ], axis=-1)

    return r_hat, theta_hat, phi_hat

def Hertz_Efield(I, dl, k, theta_grid, phi_grid, r_grid):
	prefac = I*dl/(4*np.pi*constants.epsilon_0)
	r_hat, theta_hat, phi_hat = spherical_unit_vectors(theta_grid, phi_grid)
	theta_grid, phi_grid, r_grid = [g[..., np.newaxis] for g in (theta_grid, phi_grid, r_grid)]	
	term1 = (k**2)/r_grid*np.sin(theta_grid)*np.exp(-1j*k*r_grid)*theta_hat
	term2 = (1/(r_grid**3)-1j*k/(r_grid**2))*(2*np.cos(theta_grid)*r_hat+np.sin(theta_grid)*theta_hat)*np.exp(-1j*k*r_grid)
	E = prefac*(term1+term2)
	return E

f = 1e9
wavelength = constants.c/f
k = 2*np.pi/wavelength
omega = 2*np.pi*f
r = np.linspace(0.5*wavelength, 10*wavelength, 30)
I = 1
dl = wavelength/50
n_theta = 181
n_phi = 361
theta = np.linspace(0, np.pi, n_theta)
phi = np.linspace(0, 2*np.pi, n_phi)

r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi,indexing='ij')

E = Hertz_Efield(I, dl, k, theta_grid, phi_grid, r_grid)


# Setup theta and phi plotting grids
THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

# Cartesian points on a unit sphere
X = np.sin(THETA) * np.cos(PHI)
Y = np.sin(THETA) * np.sin(PHI)
Z = np.cos(THETA)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Fix camera view
ax.view_init(elev=30, azim=30)
ax.set_xlim([-10*wavelength,10*wavelength])
ax.set_ylim([-10*wavelength, 10*wavelength])
ax.set_zlim([-10*wavelength, 10*wavelength])

# Empty surface to update
surf = [ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(np.zeros_like(X)), rstride=1, cstride=1, linewidth=0, antialiased=False)]


# Build animation
anim = animation.FuncAnimation(fig, update, frames=len(r), interval=100, blit=False,
                                fargs=(E, r, X, Y, Z, wavelength, ax, surf))

#anim.save('/home/jhirst/MoM_3D/field_animation.gif', writer='pillow', fps=10)
plt.show()