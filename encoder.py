import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """Implementation of Graph Convolutional Network (GCN) layer.
    Attributes:
      in_dim: Input dimensionality of the layer.
      out_dim: Output dimensionality of the layer.
      activation: Activation function to use for the final representations.
    """

    def __init__(self, in_dim, out_dim, activation=None):
        """Initializes the layer with specified parameters."""
        super(GCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, A_norm, X):
        """
        Computes GCN representations according to input features and input graph.
        :param A_norm: normalized (n*n) sparse graph adjacency matrix
        :param X: (n*in_dim) node feature matrix
        :return: An (n*out_dim) node representation matrix
        """
        assert isinstance(X, torch.Tensor)
        assert isinstance(A_norm, torch.sparse.Tensor)

        output = torch.matmul(X, self.weight)
        output = torch.spmm(A_norm, output) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dims):
        super(GCN, self).__init__()

        self.encoder = nn.ModuleList()
        hid_dims = [in_dim] + list(hid_dims)
        if len(hid_dims) > 2:
            for i in range(len(hid_dims) - 2):
                self.encoder.append(GCNConv(hid_dims[i], hid_dims[i + 1], activation=F.selu))
        self.encoder.append(GCNConv(hid_dims[-2], hid_dims[-1], activation=None))

    def embed(self, A_norm, X):
        Z = X
        for layer in self.encoder:
            Z = layer(A_norm, Z)
        Z = F.normalize(Z, p=2, dim=1)
        return Z

    def forward(self, A_norm1, A_norm2, X1, X2):
        # Z1 and Z2 have been normalized.
        Z1 = self.embed(A_norm1, X1)
        Z2 = self.embed(A_norm2, X2)

        return Z1, Z2
