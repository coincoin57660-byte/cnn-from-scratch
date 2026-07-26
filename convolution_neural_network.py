'''
Docstring for Python.IA.CNN.MNIST - number recongnition.convolution_neural_network

To do list :
- Backpropagation de la Convolution

- Optimisation (
    Vectorisation partielle
    im2col ou einsum, transformer la convolution en matrice x matrice
    voir peut être GPU
)
'''

import numpy as np
import os

import time
import matplotlib.pyplot as plt


base_dir = os.path.dirname(__file__)

data_path = os.path.join(base_dir, 'dataset.npz')
data = np.load(data_path)
# 60 000 data d'entrainement
X = data['X'] / 255  # Normalisation
y = data['y']

data_path_test = os.path.join(base_dir, 'dataset_test.npz')

data_test = np.load(data_path_test)
# 10 000 data de test
X_test = data_test['X'] / 255  # Normalisation
y_test = data_test['y']

print('Data load successfuly...\n')



class Softmax:
    def forward(self, x):
        x = x - np.max(x, axis = 1, keepdims = True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis = 1, keepdims = True)

    def backward(self, dout):
        # à condition que la loss calcule : dout = probs - target
        return dout


class ReLU:
    def forward(self, x):
        # appliqué à chaque élément
        # mask shape : (N, C, H, W)
        self.mask = x > 0
        return x * self.mask
        # (N, C, H, W) -> (N, C, H, W)

    def backward(self, dout):
        return dout * self.mask
        # (N, C, H, W) -> (N, C, H, W)

class Flatten():
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
        # (N, C, H, W) -> (N, CxHxW)

    def backward(self, dout):
        return dout.reshape(self.input_shape)
        # (N, CxHxW) -> (N, C, H, W)



class MaxPoolingLayer:
    def __init__(self, pool_H: int, pool_W: int) -> None:
        self.pool_W = pool_W
        self.pool_H = pool_H
        # marche implicitement uniquement pour kernel_size = 3 et pooling = 2

    def forward(self, x):
        self.x = x

        N, C, H, W = x.shape
        H_out = H // self.pool_H
        W_out = W // self.pool_W

        # tenseur de zeros pour aprés le remplir avec le pooling
        out = np.zeros((N, C, H_out, W_out))
        # tenseur pour sauvgarder la location de la valeur maximum dans chaque patch
        self.argmax = np.zeros((N, C, H_out, W_out), dtype = int)

        # boucle pour chaque image dans batch
        for n in range(N):
            # boucle sur chaque feature maps
            for c in range(C):
                # boucle sur chaque position (i, j)
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.pool_H
                        w_start = j * self.pool_W

                        patch = x[
                            n,  # le batch
                            c,  # la feature map
                            h_start: h_start + self.pool_H,
                            w_start: w_start + self.pool_W
                        ]

                        # max local et indice
                        index = np.argmax(patch)
                        out[n, c, i, j] = patch.flat[index]  # valeur max
                        self.argmax[n, c, i, j] = index  # indice local
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        H_out = dout.shape[2]
        W_out = dout.shape[3]

        # tenseur de la même shape que l'entrée du forward
        dx = np.zeros_like(self.x)

        # boucle pour chaque image dans batch
        for n in range(N):
            # boucle sur chaque feature maps
            for c in range(C):
                # boucle sur chaque position (i, j)
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.pool_H
                        w_start = j * self.pool_W

                        # récupère l'indice du max
                        index = self.argmax[n, c, i, j]

                        # crée une copy du patch, mais toute modification dans cette copy impact dx
                        patch = dx[
                            n,
                            c,
                            h_start: h_start + self.pool_H,
                            w_start: w_start + self.pool_W
                        ]
                        # remplace la valeur de la localisation de la valeur max par la valeur en question
                        patch.flat[index] = dout[n, c, i, j]

        return dx


class DenseLayer:
    def __init__(self, input_size: int, output_size: int) -> None:
        # initialisation des poids spécialement pour le ReLU, 'He'
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.b = np.zeros(output_size)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        # W[i,j] relie entrée i -> sortie j
        # - de combien l’entrée i était activée : x
        # - de combien la sortie j influence la loss : dout
        # gradient d’un poids =
        #   activité AVANT la connexion
        #             ×
        #   erreur APRÈS la connexion
        # x = (batch, in), dout = (batch, out)
        # dW = x.T @ dout
        # (in, batch) @ (batch, out) = (in, out)
        self.dW = self.x.T @ dout
        # (batch, out) -> (out,)
        self.db = np.sum(dout, axis = 0)
        dx = dout @ self.W.T
        return dx
        # (batch, dx)

    def update(self, lr: float) -> None:
        # update les poids et bias aprés le backward de tout le network
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Convolution2DLayer:
    def __init__(self, out_channels: int, in_channels: int, kernel_size: int, padding: int) -> None:
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding

        fan_in = kernel_size * kernel_size* in_channels
        std = np.sqrt(2.0 / fan_in)
        # poids des kernels (tenseur 4D)
        # 4 dimentions, tenseur
        # -> 2 dimentions pour le kernel
        # -> une 3ème pour les différents channels d'entrée
        # -> et une 4ème dimentions pour les différnete feature mpas
        self.W = np.random.normal(
            loc = 0.0,
            scale = std,
            size = (out_channels, in_channels, kernel_size, kernel_size)
        ) # (C_out, C_int, k, k)
        # bias initialisés à 0
        self.b = np.zeros(out_channels) # (C_out,)

    def forward(self, x):
        self.x = x
        # entrée x : (N, C_in, H, W)
        # sortiee out : (N, C_out, H, W)

        # gestion du padding
        if self.padding > 0:
            self.x_padded = np.pad(
                x,
                # (0, 0) -> pas de padding sur le batch et les channels, seulement sur Hauteur / Largeur
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode = 'constant'
            )
        else:
            self.x_padded = x

        # extraction des tailles
        C_out = self.out_channels
        k = self.kernel_size

        N, _, H, W = self.x.shape

        # création de la sortie, la convolution va remplir valeur par valuer
        out = np.zeros((N, C_out, H, W))

        # boucle pour chaque image dans batch
        for n in range(N):
            # boucler sur les kernels (feature maps)
            for c_out in range(C_out):
                # chaque kernel -> 1 feature map
                # boucler sur chaque position (i, j)
                for i in range(H):
                    for j in range(W):
                        # chaque (i, j) correspond à un pixel de sortie
                        # extraction du patch local
                        patch = self.x_padded[n, :, i:i+k, j:j+k]
                        # shape du patch : (N, C_in, k, k), la taille du kernel

                        # sommes pondéré : produit + somme + bias
                        out[n, c_out, i, j] = (
                            np.sum(patch * self.W[c_out])
                            + self.b[c_out]
                        )
                        # -> somme(pixels x poids) + bias

        return out

    def backward(self, dout):
        # x : (N, Cin, Hin, Win)
        # W : (Cout, Cin, Kw, Kw)
        # out : (N, Cout, Hout, Wout)
        # dout : (N, Cout, Hout, Wout)
        # -> dx : (N, Cin, Hin, Win)

        # forward = un patch de x influence UNE valeur de out
        # backward = UNE valeur de dout influence TOUT LE PATCH de x

        # but = rendre à chacun proportionnellement à ce qu'il a contribué

        N, C, H, W = self.x.shape
        F, _, Kh, Kw = self.W.shape
        _, _, Hout, Wout = dout.shape

        # initialisation
        dx_padded = np.zeros_like(self.x_padded)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros(F)

        # boucle sur chaque batch
        for n in range(N):
            # boucle sur chaque featue maps
            for f in range(F):
                # pour chaque position (i, j)
                for i in range(Hout):
                    for j in range(Wout):
                        # définie le patch affécté par le kenel
                        h_start = i
                        h_end = h_start + Kh
                        w_start = j
                        w_end = w_start + Kw

                        # récupération du patch
                        patch = self.x_padded[
                            n,
                            :,
                            h_start:h_end,
                            w_start:w_end
                        ]

                        # prend le gradient
                        grad = dout[n, f, i, j]

                        # calcule de dW
                        # donne proportionnellement la distribution du gradient
                        self.dW[f] += patch * grad

                        # calcule de dx
                        dx_padded[n, :, h_start: h_end, w_start: w_end] += self.W[f] * grad

                        # calcule de db
                        self.db[f] += grad

        # fait la moyen sur l'ensemble des batchs
        self.dW /= N
        self.db /= N

        # retour x dans sa forme non padded
        dx = dx_padded[:, :, self.padding: -self.padding, self.padding: -self.padding]
        return dx

    def update(self, lr: float) -> None:
        # update les kernels et bias aprés le backward de tout le network
        self.W -= lr * self.dW
        self.b -= lr * self.db




# Convention standard
# (N, C, H, W) -> (batch, channels, height, width)
network = [
    # input : 28x28x1, 784 valeurs
    Convolution2DLayer(out_channels = 32, in_channels = 1, kernel_size = 3, padding = 1),
    ReLU(),
    # output : 28x28x32, 25088 valeurs
    # shape : (N, C, H, W) => (N, 32, 28, 28)
    # 320 paramètres / poids + bias
    # 1 image -> 32 kernels -> 32 feature maps
    # 28x28x32x9 = 226 000 opérations

    MaxPoolingLayer(pool_H = 2, pool_W = 2),
    # output : 14x14x32, 6272 valeurs
    # shape : (N, C, H, W) => (N, 32, 14, 14)

    Convolution2DLayer(out_channels = 64, in_channels = 32, kernel_size = 3, padding = 1),
    ReLU(),
    # output : 14x14x64, 12544 valeurs
    # shape : (N, C, H, W) => (N, 64, 14, 14)
    # - prend le patch 3x3 dans chaque feature maps
    # - 32 sommes pondérées + bias => 1 valeur
    # 64 kernels -> 64 feature maps
    # 28x28x64x(9x32) = 14 450 688 opérations x batch_size
    # 18 496 paramètres / poids + bias
    # plus rapide que la première convolution car 4x moins de tours de boucle sur l'espace

    MaxPoolingLayer(pool_H = 2, pool_W = 2),
    # output : 7x7x64, 3136 valeurs
    # shape : (N, C, H, W) => (N, 64, 7, 7)

    Flatten(),
    # (N, 64, 7, 7) -> (N, 3136)
    # output : vecteur de 3136 dimentions
    # shape : (N, 3136)

    DenseLayer(input_size = 3136, output_size = 512),
    ReLU(),
    # output : 512 valeurs
    # shape : (N, 512)
    # 1 606 144 paramètres / poids + bias

    DenseLayer(input_size = 512, output_size = 128),
    ReLU(),
    # output : 128 valeurs
    # shape : (N, 128)
    # 65 664 paramètres / poids / bias

    DenseLayer(input_size = 128, output_size = 10),
    # 1 290 paramètres / poids + bias
    Softmax()
    # output : renvoit vecteur de probabilité des 10 calsses normalisé entre 0 et 1
    # shape : (N, 10)
    # représentation sous la forme : [[p0, p1, p2, p3, p4, p5, p6, p7, p8, p9], ..., N fois ,...]
]
# environ 16 000 000 d'opérations par image pour forward



def forward(network, x):
    for layer in network:
        x = layer.forward(x)
    return x

def backward(network, grad):
    for layer in reversed(network):
        grad = layer.backward(grad)

def update(network, lr: float) -> None:
    for layer in network:
        if hasattr(layer, 'update'):
            layer.update(lr)



def forward_data(network, x):
    datas = []

    for layer in network:
        shape_x = x.shape
        temp_time = time.time()
        x = layer.forward(x)
        datas.append((layer.__class__.__name__, time.time() - temp_time, shape_x, x.shape))

    return x, datas

def backward_data(network, grad):
    datas = []

    for layer in reversed(network):
        shape_grad = grad.shape
        temp_time = time.time()
        grad = layer.backward(grad)
        datas.append((layer.__class__.__name__, time.time() - temp_time, shape_grad, grad.shape))

    return datas



def debuge_analyse_shape(network, batch_size: int):
    print()
    print(f'Forward')
    print()

    print(f'Estimated time : {20 / 128 * batch_size:.2f}s\n')

    time_start = time.time()


    y_pred, datas = forward_data(network = network, x = x_batch)

    for name, data, inshape, outshape in datas:
        print(f'|  {name:<23}  |  time : {data:<10.6f}  |  input shape : {inshape!s:<17}  |  output shape : {outshape!s:<17}  |')

    print(f'\nTotal time : {time.time() - time_start:.2f}s')


    print()
    print(f'\nBackward\n')
    print(f'Estimated time : {26 / 128 * batch_size:.2f}s\n')


    time_start = time.time()

    datas = backward_data(network = network, grad = y_pred)

    for name, data, inshape, outshape in datas:
        print(f'|  {name:<23}  |  time : {data:<10.6f}  |  input shape : {inshape!s:<17}  |  output shape : {outshape!s:<17}  |')

    print(f'\nTotal time : {time.time() - time_start:.2f}s')


def train(network, lr = 0.005, batch_size = 8, nbr_itération = 10):
    global x_batch
    batch_size = batch_size
    lr = lr
    nbr_itération = nbr_itération

    # batch de n images
    x_batch = X[:batch_size]  # (n, 784)
    x_batch = x_batch.reshape(-1, 28, 28)  # (n, 28, 28)
    x_batch = x_batch[:, None, :, :]  # (n, 1, 28, 28)
    y_batch = y[:batch_size]

    start_time = time.time()

    # X2 sur les PC du Lycée
    time_ = (36 / 10 / 10) * batch_size * nbr_itération
    if time_ < 60:  # en secondes
        print(f'Estmated time : {time_:.2f}s\n')
    elif time_ < 60 * 60:  # en minutes
        print(f'Estmated time : {time_ / 60:.2f}s\n')
    else:  # en heures
        print(f'Estmated time : {time_ / 60 / 60:.2f}h\n')

    for i in range(nbr_itération):

        y_pred = forward(network = network, x = x_batch)

        # loss
        loss = -np.mean(np.log(
            y_pred[np.arange(batch_size), y_batch] + 1e-9
        ))

        # gradient softmax + CE
        gradient = y_pred.copy()
        gradient[np.arange(batch_size), y_batch] -= 1
        # moyenne du gradient
        gradient /= batch_size

        backward(network = network, grad = gradient)

        update(network = network, lr = lr)

        if i % (nbr_itération / 10) == 0:
            print(f'Loss : {loss}')

    time__ = time.time() - start_time
    if time__ < 60:  # en secondes
        print(f'\nTime : {time_:.2f}s')
    elif time__ < 60 * 60:  # en minutes
        print(f'\nTime : {time_ / 60:.2f}min')
    else:  # en heures
        print(f'\nTime : {time_ / 60 / 60:.2f}min')


# script principal
if __name__ == '__main__':

    train(network = network, lr = 0.005, batch_size = 1, nbr_itération = 100)
    # 100 images forward + backward en 36s

    print(f'\nProgramme executed successfuly')