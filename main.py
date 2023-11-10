import warnings

warnings.filterwarnings('ignore')

import torch
from encoder import GCN
from torch.optim import Adam
from utils import get_logger, fix_seed, evaluate, normalize, augment, load_data

if __name__ == "__main__":
    # Load dataset
    dataset = 'Cora'
    A, X, Y = load_data(dataset)

    # Set hyper-parameters
    device = 'cuda:0'
    in_dim = X.shape[1]
    # Copy from params.txt
    epochs = 200
    lam, gam = 1., 1.
    s, tau = 0.6, 0.1
    hid_dims = [256, 64]
    lr, wd = 1e-03, 1e-05
    pd1, pd2, pm1, pm2 = 0.3, 0.3, 0.2, 0.2

    logger = get_logger(f'./{dataset}_NS4GC.log')
    params = {
        'epochs': epochs,
        'lr, wd': (lr, wd),
        's, tau': (s, tau),
        'hid_dims': hid_dims,
        'lam, gam': (lam, gam),
        'drop_rate': (pd1, pd2, pm1, pm2)
    }
    logger.info(str(params))

    fix_seed(0)  # for reproduction
    encoder = GCN(in_dim, hid_dims)
    encoder = encoder.to(device)
    A = A.to(device)
    X = X.to(device)

    src, dst = A._indices()
    mask = torch.full(A.size(), True, device=A.device)
    mask[src, dst] = False
    mask.fill_diagonal_(False)

    # Pre-train
    optimizer = Adam(encoder.parameters(), lr=lr, weight_decay=wd)
    for epoch in range(epochs):
        encoder.train()
        A1, X1 = augment(A, X, pd1, pm1)
        A2, X2 = augment(A, X, pd2, pm2)
        A_norm1 = normalize(A1, add_self_loops=True)
        A_norm2 = normalize(A2, add_self_loops=True)
        Z1, Z2 = encoder(A_norm1, A_norm2, X1, X2)

        # Z1 and Z2 have been normalized.
        S = Z1 @ Z2.T

        loss_ali = - torch.diag(S).mean()

        loss_nei = - S[src, dst].mean()

        S = torch.masked_select(S, mask)
        S = torch.sigmoid((S - s) / tau)
        loss_spa = S.mean()

        loss = loss_ali + lam * loss_nei + gam * loss_spa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'E:{epoch} Loss:{loss.item():.4f} ALI:{loss_ali.item():.4f} '
              f'NEI:{loss_nei.item():.4f} SPA:{loss_spa.item():.4f}')

    # Clustering
    encoder.eval()
    A_norm = normalize(A, add_self_loops=True)
    with torch.no_grad():
        Z = encoder.embed(A_norm, X)
    evaluate(Z.cpu().numpy(), Y.cpu().numpy(), logger)

    logger.info('')
