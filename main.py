import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image
# Scalrt
from sklearn.preprocessing import StandardScaler

# Carregar o conjunto de dados Digits
digits = load_digits()

# Obter os dados e os rótulos
X = digits.data
y = digits.target

# Informações sobre o conjunto de dados
n_samples, n_features = X.shape
n_digits = len(np.unique(y))

## 1. Exploração dos Dados:
print("Número de exemplos: %d" % n_samples)
print("Dimensões das imagens: %d x %d" % (n_features, int(np.sqrt(n_features))))
print("Distribuição dos dígitos:")
print(np.bincount(y))

# Exibir alguns exemplos de imagens
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[y == i][0].reshape(8, 8), cmap='gray')
    plt.title("Dígito %d" % i)
    plt.axis('off')

plt.tight_layout()
# Salvando a figura
plt.savefig('figura_dataset_digits.png')
plt.show()

# 2. Pré-processamento dos Dados:
# Pré-processamento dos dados com PCA

# Aplicar o escalamento dos dados
"""scaler = StandardScaler()
X = scaler.fit_transform(X)"""


n_components = 2  # Manter todos os 64 componentes principais
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

print("X_pca.shape:", X_pca.shape)
print("X_pca[0]:", X_pca[0])

print("X.shape:", X.shape)
print("X[0]:", X[0])

# Plot dos dados após o pré-processamento
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Digits Dataset após Pré-processamento')
plt.colorbar()
# Salvando a figura
plt.savefig('figura_dataset_digits_pca.png')
plt.show()

# 3. Implementação do K-means:

# Implementação do K-means
'''
kmeans = KMeans(n_clusters=10, init='k-means++', random_state=42)
kmeans.fit(X_pca)
labels = kmeans.labels_
'''


# Implementação do K-means
kmeans = KMeans(init="k-means++", n_clusters=n_digits, random_state=42, n_init=10)
kmeans.fit(X_pca)
labels = kmeans.labels_

# Plot dos clusters resultantes
plt.figure(figsize=(10, 6))
colors = ['pink', 'g', 'r', 'c', 'm', 'y', 'brown', 'orange', 'purple', 'gray']
for i in range(n_digits):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], color=colors[i], label=str(i))

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Resultado do K-means - Digits Dataset')

# Plotar os centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroides', marker='*')

plt.legend()
# Salvando a figura
plt.savefig('figura_dataset_digits_kmeans.png')
plt.show()

# 4. Avaliação do Modelo:
# Avaliação do modelo
silhouette_avg = silhouette_score(X_pca, labels)  # Índice de Silhouette
print("Índice de Silhouette:", silhouette_avg)

soma = kmeans.inertia_  # Soma das distâncias quadradas intra-cluster
print("Soma das distâncias quadradas intra-cluster:", soma)


# 5. Teste com Dados Próprios:
# Caminho para as imagens de teste
path_imagens = 'imagens'

path_imagens = [os.path.join(path_imagens, imagem) for imagem in os.listdir(path_imagens)]

new_digits = []
for imagem in path_imagens:
    image = Image.open(imagem)
    image = image.convert('L')
    image = image.resize((8, 8))
    image = np.array(image)
    image = image.flatten()
    image = 255 - image
    image = image.reshape(8, 8)
    image = Image.fromarray(image.astype('uint8'))
    image = np.array(image)
    image = image.flatten()
    new_digits.append(image)

# Scaler
# new_digits = scaler.transform(new_digits)
new_digits_pca = pca.transform(new_digits)
predicted_labels = kmeans.predict(new_digits_pca)

print("Dígitos manuscritos pelo usuário:")
print(predicted_labels)

# Plotar os dígitos manuscritos pelo usuário
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(new_digits[i].reshape(8, 8), cmap='gray')
    plt.title("Dígito %d" % predicted_labels[i])
    plt.axis('off')

plt.tight_layout()
# Salvando a figura
plt.savefig('figura_dataset_digits_manuscrito.png')
plt.show()