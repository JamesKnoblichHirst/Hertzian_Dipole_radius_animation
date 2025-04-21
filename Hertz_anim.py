import numpy as np
from scipy import constants
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
from numba import njit

def update(frame_idx, E, r, X, Y, Z, wavelength, ax, surf):
    ax.clear()
    ax.set_xlim([-10*wavelength,10*wavelength])
    ax.set_ylim([-10*wavelength,10*wavelength])
    ax.set_zlim([-10*wavelength,10*wavelength])
    ax.view_init(elev=30, azim=30)
    ax.set_axis_off()

    E_slice = E[frame_idx, :, :, :]
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
    ], axis=-1)

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

@njit
def Hertz_Efield(I, dl, k, theta_grid, phi_grid, r_grid):
    prefac = I * dl / (4 * np.pi * 8.854187817e-12)  # constants.epsilon_0 hardcoded
    sin_theta = np.sin(theta_grid)
    cos_theta = np.cos(theta_grid)
    sin_phi = np.sin(phi_grid)
    cos_phi = np.cos(phi_grid)

    r_hat = np.zeros(theta_grid.shape + (3,))
    theta_hat = np.zeros(theta_grid.shape + (3,))
    phi_hat = np.zeros(theta_grid.shape + (3,))

    r_hat[..., 0] = sin_theta * cos_phi
    r_hat[..., 1] = sin_theta * sin_phi
    r_hat[..., 2] = cos_theta

    theta_hat[..., 0] = cos_theta * cos_phi
    theta_hat[..., 1] = cos_theta * sin_phi
    theta_hat[..., 2] = -sin_theta

    phi_hat[..., 0] = -sin_phi
    phi_hat[..., 1] = cos_phi
    phi_hat[..., 2] = 0.0

    term1 = (k ** 2) / r_grid * np.sin(theta_grid) * np.exp(-1j * k * r_grid)[:, :, np.newaxis] * theta_hat
    term2 = (1 / r_grid ** 3 - 1j * k / r_grid ** 2) * (
        2 * np.cos(theta_grid) * r_hat + np.sin(theta_grid) * theta_hat
    ) * np.exp(-1j * k * r_grid)[:, :, np.newaxis]

    E = prefac * (term1 + term2)

    return E


def run_simulation(save_path=None, anim_interval=100, n_points=30):
    try:
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not save_path.suffix == ".gif":
                raise ValueError("Save path must end with .gif")

        if anim_interval <= 0:
            raise ValueError("Animation interval must be positive.")

        if n_points < 2 or n_points > 1000:
            raise ValueError("Number of points must be between 2 and 1000.")

        f = 1e9
        wavelength = constants.c / f
        k = 2 * np.pi / wavelength
        r = np.linspace(0.5*wavelength, 10*wavelength, n_points)
        I = 1
        dl = wavelength / 50
        n_theta = 181
        n_phi = 361
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi, indexing='ij')

        E = Hertz_Efield(I, dl, k, theta_grid, phi_grid, r_grid)

        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        X = np.sin(THETA) * np.cos(PHI)
        Y = np.sin(THETA) * np.sin(PHI)
        Z = np.cos(THETA)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=30)
        ax.set_xlim([-10*wavelength,10*wavelength])
        ax.set_ylim([-10*wavelength,10*wavelength])
        ax.set_zlim([-10*wavelength,10*wavelength])

        surf = [ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(np.zeros_like(X)),
                                rstride=1, cstride=1, linewidth=0, antialiased=False)]

        anim = animation.FuncAnimation(fig, update, frames=len(r), interval=anim_interval, blit=False,
                                        fargs=(E, r, X, Y, Z, wavelength, ax, surf))

        if save_path is not None:
            anim.save(save_path, writer='pillow', fps=int(1000/anim_interval))
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate an animated Hertzian dipole field.")
    parser.add_argument('--save_path', type=str, default=None,
                        help="Path to save the output GIF. If not given, just displays animation.")
    parser.add_argument('--interval', type=int, default=100,
                        help="Animation speed in milliseconds between frames. Default is 100ms.")
    parser.add_argument('--points', type=int, default=30,
                        help="Number of radial points to simulate. Default is 30.")

    args = parser.parse_args()

    run_simulation(save_path=args.save_path, anim_interval=args.interval, n_points=args.points)

if __name__ == "__main__":
    main()
