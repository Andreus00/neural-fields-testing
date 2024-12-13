import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

# Definizione dei parametri della gaussiana
mu_x, mu_y, mu_z = 0, 0, 0  # Medie
sigma_x, sigma_y, sigma_z = 0.4, 1.4, 0.2  # Deviazioni standard

plane_curvature_factor = 1

def generate_gaussian(pos, scale):

    # Creazione della griglia di punti per l'ellissoide
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    u, v = np.meshgrid(u, v)

    # Parametrizzazione dell'ellissoide
    x = scale[0] * np.sin(v) * np.cos(u)
    y = scale[1] * np.sin(v) * np.sin(u)
    z = scale[2] * np.cos(v)
    
    r = pos[1] * 180 + 90 if pos[2] <= 0 else -pos[1] * 180 + 90
    rotation = R.from_euler('xyz', [r, 0, 0], degrees=True)
    print('asin', np.asin(pos[1]), 'pos', print(pos[1]))
    rotated_points = rotation.apply(np.array([x.flatten(), y.flatten(), z.flatten()]).T).T
    x = rotated_points[0].reshape(x.shape)
    y = rotated_points[1].reshape(y.shape)
    z = rotated_points[2].reshape(z.shape)
    x += pos[0]
    y += pos[1]
    z += pos[2]
    return x, y, z, r

gaussians = []
positions = []
rotations = []
scales = []

# Creazione della griglia per il piano curvo
plane_x = np.linspace(-1.5, 1.5, 100)
plane_y = np.linspace(-1.5, 1.5, 100)
plane_x, plane_y = np.meshgrid(plane_x, plane_y)

plane_z = plane_curvature_factor * np.sin(np.pi * plane_y)

for i in range(100):
    scale = np.random.randint(2, 5, size=(3,)) * 0.1
    pos = np.random.randint(-10, 10, size=(3,)) * 0.1
    scale[2] *= 0.5
    pos[2] = plane_curvature_factor * np.sin(np.pi * pos[1])
    print('pos2', pos[2])
    *coords, r = generate_gaussian(pos, scale)
    gaussians.append(coords)
    rotations.append(r)
    positions.append(pos)
    scales.append(scale)


# Funzione per aggiornare l'animazione
def update(frame):
    ax.cla()
    
    # Rotazione dinamica
    rotation = R.from_euler('xyz', [0, 0, frame], degrees=True)
    # rotation = R.from_euler('xyz', [0, 0, 0], degrees=True)
    for (x, y, z), position, scale, rot in zip(gaussians, positions, scales, rotations):
        rotated_points = rotation.apply(np.array([x.flatten(), y.flatten(), z.flatten()]).T).T
        x_rotated = rotated_points[0].reshape(x.shape)
        y_rotated = rotated_points[1].reshape(y.shape)
        z_rotated = rotated_points[2].reshape(z.shape)

        plane_points = np.array([plane_x.flatten(), plane_y.flatten(), plane_z.flatten()])
        rotated_plane = rotation.apply(plane_points.T).T
        plane_x_rotated = rotated_plane[0].reshape(plane_x.shape)
        plane_y_rotated = rotated_plane[1].reshape(plane_y.shape)
        plane_z_rotated = rotated_plane[2].reshape(plane_z.shape)

        # Plot dell'ellissoide ruotato
        ax.plot_surface(x_rotated, y_rotated, z_rotated, cmap='viridis', edgecolor='none', alpha=0.7)

        # Vettori principali
        # principal_vectors = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, scale[2]]])
        # rotation2 = R.from_euler('xyz', [rot, 0, 0], degrees=True)
        # principal_vectors = rotation2.apply(principal_vectors)
        # principal_vectors = rotation.apply(principal_vectors)
        # origin = [position[0], position[1], position[2]]
        # origin = rotation2.apply(origin)
        # origin = rotation.apply(origin)
        # origin = [position[0], position[1], position[2]]
        # for vec in principal_vectors:
        #     ax.quiver(*origin, *vec, color='#FF0000', linewidth=3, arrow_length_ratio=0.2)

    # Plot del piano curvo
    ax.plot_surface(plane_x_rotated, plane_y_rotated, plane_z_rotated, cmap='coolwarm', edgecolor='none', alpha=0.5)

    # Etichette e limiti
    ax.set_title('3D Gaussians on a plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1, 1)

# Creazione della figura e dell'animazione
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)


# Salvataggio o visualizzazione dell'animazione
plt.show()
