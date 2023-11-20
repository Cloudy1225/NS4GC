import os
import torch
import random
import logging
import numpy as np
from munkres import Munkres
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, \
    normalized_mutual_info_score, adjusted_rand_score
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, CoraFullDataset, \
    WikiCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorCSDataset


def load_data(dataset):
    if dataset == 'Cora':
        G = CoraGraphDataset()[0]
    elif dataset == 'Citeseer':
        G = CiteseerGraphDataset()[0]
    elif dataset == 'Pubmed':
        G = PubmedGraphDataset()[0]
    elif dataset == 'CoraFull':
        G = CoraFullDataset()[0]
    elif dataset == 'WikiCS':
        G = WikiCSDataset()[0]
    elif dataset == 'Photo':
        G = AmazonCoBuyPhotoDataset()[0]
    elif dataset == 'Computer':
        G = AmazonCoBuyComputerDataset()[0]
    elif dataset == 'CoauthorCS':
        G = CoauthorCSDataset()[0]
    else:
        G = CoraGraphDataset()[0]

    A = G.adj_external()  # torch.sparse.Tensor
    X = G.ndata.pop('feat')  # torch.Tensor
    Y = G.ndata.pop('label')  # torch.Tensor
    return A, X, Y


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def metrics(y_true, y_pred):
    """
    Evaluate the clustering performance.
    :param y_true: ground truth
    :param y_pred: prediction
    :returns acc, nmi, ari, f1:
    - accuracy
    - normalized mutual information
    - adjust rand index
    - f1 score
    """
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        print('error')
        return
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')

    return acc, nmi, ari, f1


def evaluate(Z: np.ndarray, Y: np.ndarray, logger=None):
    """Evaluate embeddings on node clustering."""
    logger = print if logger is None else logger.info

    n_clusters = np.unique(Y).shape[0]
    ACCs = []
    NMIs = []
    ARIs = []
    F1s = []

    for i in range(20):
        fix_seed(i)
        kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
        Y_ = kmeans.fit_predict(Z)
        acc, nmi, ari, f1 = metrics(Y, Y_)
        ACCs.append(acc)
        NMIs.append(nmi)
        ARIs.append(ari)
        F1s.append(f1)
    ACCs = np.array(ACCs)
    NMIs = np.array(NMIs)
    ARIs = np.array(ARIs)
    F1s = np.array(F1s)
    acc_mean = ACCs.mean() * 100
    acc_std = ACCs.std() * 100
    nmi_mean = NMIs.mean() * 100
    nmi_std = NMIs.std() * 100
    ari_mean = ARIs.mean() * 100
    ari_std = ARIs.std() * 100
    f1_mean = F1s.mean() * 100
    f1_std = F1s.std() * 100
    s = f"ACC={acc_mean:.2f}+-{acc_std:.2f}, NMI={nmi_mean:.2f}+-{nmi_std:.2f}, " \
        f"ARI={ari_mean:.2f}+-{ari_std:.2f}, F1={f1_mean:.2f}+-{f1_std:.2f}"
    logger(s)


def get_logger(filename, verbosity=1, name=None, mode='a'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# ---------- Graph Augmentation ----------
def augment(A: torch.sparse.Tensor, X: torch.Tensor,
            edge_mask_rate: float, feat_drop_rate: float):
    A = drop_edge(A, edge_mask_rate)
    X = mask_feat(X, feat_drop_rate)

    return A, X


def mask_feat(X: torch.Tensor, mask_prob: float):
    drop_mask = (
            torch.empty((X.size(1),), dtype=torch.float32, device=X.device).uniform_()
            < mask_prob
    )
    X = X.clone()
    X[:, drop_mask] = 0

    return X


def drop_edge(A: torch.sparse.Tensor, drop_prob: float):
    n_edges = A._nnz()
    mask_rates = torch.full((n_edges,), fill_value=drop_prob,
                            dtype=torch.float)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)

    E = A._indices()
    V = A._values()

    E = E[:, mask_idx]
    V = V[mask_idx]
    A = torch.sparse_coo_tensor(E, V, A.shape, device=A.device)

    return A


# ---------- Graph Normalization ----------
def add_self_loop(A: torch.sparse.Tensor):
    return A + sparse_identity(A.shape[0], device=A.device)


def normalize(A: torch.sparse.Tensor, add_self_loops=True, returnA=False):
    """Normalized the graph's adjacency matrix in the torch.sparse.Tensor format"""
    if add_self_loops:
        A_hat = add_self_loop(A)
    else:
        A_hat = A

    D_hat_invsqrt = torch.sparse.sum(A_hat, dim=0).to_dense() ** -0.5
    D_hat_invsqrt[D_hat_invsqrt == torch.inf] = 0
    D_hat_invsqrt = sparse_diag(D_hat_invsqrt)
    A_norm = D_hat_invsqrt @ A_hat @ D_hat_invsqrt
    if returnA:
        return A_hat, A_norm
    else:
        return A_norm


def sparse_identity(dim, device):
    indices = torch.arange(dim).unsqueeze(0).repeat(2, 1)
    values = torch.ones(dim)
    identity_matrix = torch.sparse_coo_tensor(indices, values,
                                              size=(dim, dim), device=device)
    return identity_matrix


def sparse_diag(V: torch.Tensor):
    size = V.size(0)
    indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
    values = V
    diagonal_matrix = torch.sparse_coo_tensor(indices, values,
                                              size=(size, size), device=V.device)
    return diagonal_matrix
