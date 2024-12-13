import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Definizione dei parametri della gaussiana
mu_x, mu_y, mu_z = 0, 0, 0  # Medie
sigma_x, sigma_y, sigma_z = 0.4, 1.4, 0.2  # Deviazioni standard
# sigma_x, sigma_y, sigma_z = 1, 1, 1  # Deviazioni standard


# Parametri di rotazione (angoli in gradi)
rot_x, rot_y, rot_z = 30, 45, 60  # Rotazione lungo gli assi X, Y, Z
# rot_x, rot_y, rot_z = 0, 0, 0  # Rotazione lungo gli assi X, Y, Z

# Creazione della griglia di punti per l'ellissoide
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)

# Parametrizzazione dell'ellissoide
x = sigma_x * np.sin(v) * np.cos(u) + mu_x
y = sigma_y * np.sin(v) * np.sin(u) + mu_y
z = sigma_z * np.cos(v) + mu_z

# Appiattimento delle coordinate per applicare la rotazione
points = np.array([x.flatten(), y.flatten(), z.flatten()])

# Creazione della matrice di rotazione
rotation = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
rotated_points = rotation.apply(points.T).T

# Ripristino della forma originaria
x_rotated = rotated_points[0].reshape(x.shape)
y_rotated = rotated_points[1].reshape(y.shape)
z_rotated = rotated_points[2].reshape(z.shape)

# Vettori principali dell'ellissoide
principal_vectors = np.array([[sigma_x, 0, 0], [0, sigma_y, 0], [0, 0, sigma_z]])
rotated_vectors = rotation.apply(principal_vectors)

# Creazione della griglia per il piano curvo
plane_x = np.linspace(-1.5, 1.5, 100)
plane_y = np.linspace(-1.5, 1.5, 100)
plane_x, plane_y = np.meshgrid(plane_x, plane_y)
plane_z = 0.3 * np.sin(0.1 * np.pi * plane_x) * np.sin(0.6 * np.pi * plane_y)

# Rotazione del piano
plane_points = np.array([plane_x.flatten(), plane_y.flatten(), plane_z.flatten()])
rotated_plane = rotation.apply(plane_points.T).T
plane_x_rotated = rotated_plane[0].reshape(plane_x.shape)
plane_y_rotated = rotated_plane[1].reshape(plane_y.shape)
plane_z_rotated = rotated_plane[2].reshape(plane_z.shape)

# Creazione del plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot dell'ellissoide ruotato
ax.plot_surface(x_rotated, y_rotated, z_rotated, cmap='viridis', edgecolor='none', alpha=0.5)


# Plot del piano curvo
ax.plot_surface(plane_x_rotated, plane_y_rotated, plane_z_rotated, cmap='coolwarm', edgecolor='none', alpha=0.5)

# Plot dei vettori principali
origin = np.array([mu_x, mu_y, mu_z])
for vec in rotated_vectors:
    ax.quiver(*origin, *vec, color='red', linewidth=4, arrow_length_ratio=0.1)

# Etichette degli assi
ax.set_title('Ellissoide della Distribuzione Gaussiana 3D (Ruotato) con Vettori')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# Mostra il grafico
plt.show()
