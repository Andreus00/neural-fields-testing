import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import matplotlib.pyplot as plt

# Definizione della rete neurale
class SDFNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=1):
        super(SDFNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Funzione per generare dati di training per una sfera
def generate_sphere_sdf_data(radius=1.0, num_points=10000):
    points = torch.rand((num_points, 3)) * 2 - 1  # Punti nell'intervallo [-1, 1] per ogni dimensione
    distances = torch.norm(points, dim=1) - radius  # Calcolo della distanza dalla sfera
    return points, distances

def generate_cube_sdf_data(center, side_length, num_points=10000):
    """
    Genera punti nello spazio e calcola la SDF per un cubo.

    Args:
        center: Centro del cubo (array/list di 3 coordinate).
        side_length: Lunghezza del lato del cubo.
        num_points: Numero di punti casuali da generare.

    Returns:
        (points, sdf_values): Tuple contenente i punti (num_points, 3) e i valori SDF (num_points,).
    """
    center = np.array(center)
    half_side = side_length / 2

    # Genera punti casuali nello spazio 3D
    points = np.random.uniform(center - 2 * half_side, center + 2 * half_side, (num_points, 3))

    # Calcola la SDF per ogni punto
    sdf_values = np.max(np.abs(points - center) - half_side, axis=1)

    # SDF positivo fuori, negativo dentro
    sdf_values[sdf_values < 0] = -np.abs(sdf_values[sdf_values < 0])
    return torch.asarray(points, dtype=torch.float), torch.asarray(sdf_values, dtype=torch.float)

def generate_pyramid_sdf_data(base_vertices, apex, num_points=10000):
    """
    Genera punti nello spazio e calcola la SDF per una piramide.

    Args:
        base_vertices: Lista di 4 vertici della base quadrata (ogni vertice è un array/list di 3 coordinate).
        apex: Vertice superiore della piramide (array/list di 3 coordinate).
        num_points: Numero di punti casuali da generare.

    Returns:
        (points, sdf_values): Tuple contenente i punti (num_points, 3) e i valori SDF (num_points,).
    """
    base_vertices = np.array(base_vertices)
    apex = np.array(apex)

    # Calcola il baricentro della base per il bounding box
    base_center = np.mean(base_vertices, axis=0)

    # Trova il bounding box
    bbox_min = np.min(np.vstack([base_vertices, apex]), axis=0)
    bbox_max = np.max(np.vstack([base_vertices, apex]), axis=0)

    # Genera punti casuali nello spazio 3D
    points = np.random.uniform(bbox_min - 1, bbox_max + 1, (num_points, 3))

    # Calcola la SDF per ogni punto
    sdf_values = np.zeros(num_points)
    for i, point in enumerate(points):
        # Distanza dalle facce
        distances = []
        for j in range(4):
            next_j = (j + 1) % 4
            distances.append(
                distance_to_triangle(point, base_vertices[j], base_vertices[next_j], apex)
            )
        distances.append(distance_to_square(point, base_vertices))
        sdf_values[i] = min(distances)

    return torch.asarray(points, dtype=torch.float), torch.asarray(sdf_values, dtype=torch.float)

def distance_to_triangle(point, v1, v2, v3):
    """
    Calcola la distanza di un punto da un triangolo definito dai vertici v1, v2, v3.
    """
    # Proietta il punto sul triangolo e calcola la distanza
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / np.linalg.norm(normal)
    proj_point = point - np.dot(point - v1, normal) * normal
    return np.linalg.norm(proj_point - point)

def distance_to_square(point, vertices):
    """
    Calcola la distanza di un punto da un quadrato definito dai suoi vertici.
    """
    distances = []
    for i in range(4):
        next_i = (i + 1) % 4
        distances.append(distance_to_triangle(point, vertices[i], vertices[next_i], vertices[0]))
    return min(distances)

# Addestramento della rete
def train_sdf_network(points, distances):
    # Iparametri
    num_epochs = 100
    batch_size = 64
    learning_rate = 1e-3

    # Generazione dei dati
    dataset = torch.utils.data.TensorDataset(points, distances)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Creazione della rete
    model = SDFNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Addestramento
    for epoch in range(num_epochs):
        for batch_points, batch_distances in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_points).squeeze(-1)
            loss = criterion(predictions, batch_distances)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    return model

# Funzione per testare la rete
def test_sdf_network(model, test_points):
    with torch.no_grad():
        sdf_values = model(test_points)
    return sdf_values

def visualize_sdf(model, grid_size=200, threshold=0.01):
    # Creazione di una griglia di punti 3D
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    z = np.linspace(-1.5, 1.5, grid_size)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    points_tensor = torch.tensor(points, dtype=torch.float32)

    # Calcolo dei valori SDF
    with torch.no_grad():
        sdf_values = model(points_tensor).numpy().squeeze()

    # Selezione dei punti vicino alla superficie (SDF ~ 0)
    surface_points = points[np.abs(sdf_values) < threshold]

    # Visualizzazione
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], s=1)
    ax.set_title("Geometria Appresa")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def merge_and_visualize_models(model1, model2, steps=10, grid_size=200, threshold=0.01):
    """
    Esegue un merge lineare tra i pesi di due modelli e visualizza la geometria risultante ad ogni step.

    Args:
        model1: Primo modello (rete neurale).
        model2: Secondo modello (rete neurale).
        steps: Numero di step per il merge lineare.
        grid_size: Dimensione della griglia per la visualizzazione.
        threshold: Valore soglia per visualizzare i punti vicini alla superficie.
    """
    # Assicurati che i modelli abbiano la stessa architettura
    assert len(list(model1.parameters())) == len(list(model2.parameters())), \
        "I modelli devono avere la stessa architettura per eseguire il merge."

    # Clona il primo modello per non modificarlo
    merged_model = SDFNetwork()

    # Itera per il numero di step
    for alpha in np.linspace(0, 1, steps):
        # Esegui il merge lineare dei pesi
        with torch.no_grad():
            for param1, param2, merged_param in zip(
                model1.parameters(), model2.parameters(), merged_model.parameters()
            ):
                merged_param.data = alpha * param1.data + (1 - alpha) * param2.data

        # Visualizza la geometria risultante
        print(f"Visualizzazione per alpha = {alpha:.2f}")
        visualize_sdf(merged_model, grid_size=grid_size, threshold=threshold)


# Esempio di esecuzione
if __name__ == "__main__":

    # Generazione dei dati
    cube_points, cube_distances = generate_cube_sdf_data([0, 0, 0], 0.1)
    print(cube_points.shape, cube_distances.shape)
    cube_sdf_model = train_sdf_network(cube_points, cube_distances)

    sphere_points, sphere_distances = generate_sphere_sdf_data()
    print(sphere_points.shape, sphere_distances.shape)
    sphere_sdf_model = train_sdf_network(sphere_points, sphere_distances)

    # visualize_sdf(cube_sdf_model)
    # visualize_sdf(sphere_sdf_model)

    merge_and_visualize_models(cube_sdf_model, sphere_sdf_model, steps=10)


