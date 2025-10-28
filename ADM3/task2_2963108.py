import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_data(random_state=2024):
    """
    Generates three clusters of 3D data with distinct means and covariances.

    Parameters:
    random_state (int): Seed for reproducibility.

    Returns:
    tuple: Combined data (3D array), labels (array indicating cluster assignments)
    """
    np.random.seed(random_state)

    # Generate three clusters with distinct means and covariance matrices
    mean1 = [10, 10, 10]
    cov1 = [[2, 1.6, 1], [1.6, 2, 0.6], [1, 0.6, 2]]
    data1 = np.random.multivariate_normal(mean1, cov1, 100)

    mean2 = [20, 20, 10]
    cov2 = [[2, -1, 0], [-1, 2, 0], [0, 0, 2]]
    data2 = np.random.multivariate_normal(mean2, cov2, 100)

    mean3 = [15, 15, 25]
    cov3 = [[2, 0, 0], [0, 2, 0], [0, 0, 10]]
    data3 = np.random.multivariate_normal(mean3, cov3, 100)

    # Combine the data and assign labels to each cluster
    data = np.vstack((data1, data2, data3))
    labels = np.array([0]*100 + [1]*100 + [2]*100)  # 0 for cluster 1, 1 for cluster 2, 2 for cluster 3
    return data, labels

# Function to standardize the dataset
def standardize_data(data):
    """
    Standardizes the dataset using StandardScaler.

    Parameters:
    data (array): Original dataset.

    Returns:
    array: Scaled dataset.
    """
    # DONE: Instantiate StandardScaler and use fit_transform to scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled

# Function to apply PCA to the dataset
def apply_pca(data_scaled, n_components):
    """
    Applies PCA to reduce dimensionality of the dataset.

    Parameters:
    data_scaled (array): Scaled dataset.
    n_components (int): Number of principal components.

    Returns:
    array: PCA transformed data.
    PCA object: The PCA object used.
    """
    # DONE: Instantiate PCA with n_components and apply PCA transformation
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    return data_pca, pca

# Function to plot the original 3D data
def plot_original_data(data, labels):
    """
    Plots the original 3D data with different colors for each cluster.

    Parameters:
    data (array): 3D dataset with combined points from all clusters.
    labels (array): Cluster labels for each data point, where each label indicates which cluster the point belongs to.
    """
    # TO DO: Use scatter plot to visualize the original data in 3D space
    fig = plt.figure(dpi=300)
    ax = plt.axes(projection='3d') # 3D projection
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=20) # Set integer labels (y) as colors
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.title("Original 3D data")
    plt.show()

# Function to plot the XY projection
def plot_xy_projection(data, labels):
    """
    Plots the XY projection of the data with colors based on cluster labels.

    Parameters:
    data (array): 3D dataset.
    labels (array): Cluster labels for each data point.
    """
    # DONE: Implement the function to plot the XY projection of the dataset
    # X, Y = index 0 and 1
    plt.figure(dpi=300)
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("XY projection")
    plt.colorbar(label='Class Label')
    plt.tight_layout()
    plt.show()

# Function to plot the XZ projection
def plot_xz_projection(data, labels):
    """
    Plots the XZ projection of the data with colors based on cluster labels.

    Parameters:
    data (array): 3D dataset.
    labels (array): Cluster labels for each data point.
    """
    # DONE: Implement the function to plot the XZ projection of the dataset
    # X, Z = index 0 and 2
    plt.figure(dpi=300)
    plt.scatter(data[:, 0], data[:, 2], c=labels, s=20)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("XZ projection")
    plt.colorbar(label='Class Label')
    plt.tight_layout()
    plt.show()


# Function to plot the YZ projection
def plot_yz_projection(data, labels):
    """
    Plots the YZ projection of the data with colors based on cluster labels.

    Parameters:
    data (array): 3D dataset.
    labels (array): Cluster labels for each data point.
    """
    # DONE: Implement the function to plot the YZ projection of the dataset
    # Y, Z = index 1 and 2
    plt.figure(dpi=300)
    plt.scatter(data[:, 1], data[:, 2], c=labels, s=20)
    plt.xlabel("y")
    plt.ylabel("z")
    plt.title("YZ projection")
    plt.colorbar(label='Class Label')
    plt.tight_layout()
    plt.show()

# Function to plot the PCA results
def plot_pca_results(data_pca, labels):
    """
    Plots the 2D PCA projection of the dataset with colors based on cluster labels.

    Parameters:
    data_pca (array): PCA-transformed dataset (2D projection).
    labels (array): Cluster labels for each data point.
    """
    # DONE: Use scatter plot to visualize the 2D projection from PCA
    plt.figure(dpi=300)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, s=20)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA 2D projection")
    plt.colorbar(label='Class Label')
    plt.show()


if __name__ == "__main__":
    np.random.seed(2024)
    data, labels = generate_data() # Generate the data and labels
    print(data.shape)
    
    # Plot original 3D data and projections
    plot_original_data(data, labels)
    plot_xy_projection(data, labels)
    plot_xz_projection(data, labels)
    plot_yz_projection(data, labels)

    # Standardize the dataset
    data_scaled = standardize_data(data)

    # DONE: Fill in appropriate value for n_components
    data_pca, pca = apply_pca(data_scaled, n_components=2)  # Apply PCA to the dataset
    
    plot_pca_results(data_pca, labels)   # Plot PCA results