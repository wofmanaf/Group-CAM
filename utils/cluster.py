import torch
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


def group_cluster(x, group=32, cluster_method='k_means'):
    # x : (torch tensor with shape [1, c, h, w])
    xs = x.detach().cpu()
    b, c, h, w = xs.shape
    xs = xs.reshape(b, c, -1).reshape(b*c, h*w)
    if cluster_method == 'k_means':
        n_cluster = KMeans(n_clusters=group, random_state=0).fit(xs)
    elif cluster_method == 'agglomerate':
        n_cluster = AgglomerativeClustering(n_clusters=group).fit(xs)
    else:
        assert NotImplementedError

    labels = n_cluster.labels_
    del xs
    return labels


def group_sum(x, n=32, cluster_method='k_means'):
    b, c, h, w = x.shape
    group_idx = group_cluster(x, group=n, cluster_method=cluster_method)
    init_masks = [torch.zeros(1, 1, h, w).to(x.device) for _ in range(n)]
    for i in range(c):
        idx = group_idx[i]
        init_masks[idx] += x[:, i, :, :].unsqueeze(1)
    return init_masks
