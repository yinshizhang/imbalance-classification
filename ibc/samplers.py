import numpy as np
from sklearn.neighbors import NearestNeighbors


# blmovgen class for binary classification
def blmovgen(x, y, n_neighbors=5, alpha=1.0, **kwargs):
    # sample strategy for each minority classes
    _, counts = np.unique(y, return_counts=True)
    sample_size = max(counts) - counts

    gen_x = []
    gen_y = []

    # generate synthetic data for each minority class
    for i, size in enumerate(sample_size):
        if size == 0:
            continue
        min_idxs = np.where(y == i)[0]
        maj_idxs = np.where(y != i)[0]

        # find nearest majority neighbors for each minority instance
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x[maj_idxs])
        _, indices = nbrs.kneighbors(x[min_idxs])

        # generate synthetic data
        for j in np.random.randint(0, len(min_idxs), size):
            min_idx = min_idxs[j]
            rand_idx = np.random.randint(1, n_neighbors)
            maj_idx1 = maj_idxs[indices[j][0]]
            maj_idx2 = maj_idxs[indices[j][rand_idx]]
            new_x = x[min_idx] + alpha * (x[maj_idx2] - x[maj_idx1])
            gen_x.append(new_x)
            gen_y.append(i)
    gen_x = np.array(gen_x)
    gen_y = np.array(gen_y)
    return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)


# admovgen class
def admovgen(x, y, m_neighbors=5, n_neighbors=5, alpha=1.0):
    # sample strategy for each minority classes
    _, counts = np.unique(y, return_counts=True)
    sample_size = max(counts) - counts

    gen_x = []
    gen_y = []

    # generate synthetic data for each minority class
    global_nbrs = NearestNeighbors(n_neighbors=m_neighbors).fit(x)
    for i, size in enumerate(sample_size):
        if size == 0:
            continue
        min_idxs = np.where(y == i)[0]
        maj_idxs = np.where(y != i)[0]

        # calculate the weights for each minority instance
        # the only difference from blmovgen
        _, min_nbrs_ids = global_nbrs.kneighbors(x[min_idxs])
        weights = np.array([(y[id] == i).sum() for id in min_nbrs_ids])

        # find nearest majority neighbors for each minority instance
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x[maj_idxs])
        _, indices = nbrs.kneighbors(x[min_idxs])

        # generate synthetic data
        for j in np.random.choice(len(min_idxs), size=size, p=weights / weights.sum()):
            min_idx = min_idxs[j]
            rand_idx = np.random.randint(1, n_neighbors)
            maj_idx1 = maj_idxs[indices[j][0]]
            maj_idx2 = maj_idxs[indices[j][rand_idx]]
            new_x = x[min_idx] + alpha * (x[maj_idx2] - x[maj_idx1])
            gen_x.append(new_x)
            gen_y.append(i)
    gen_x = np.array(gen_x)
    gen_y = np.array(gen_y)
    return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)


# base_adasyn class
def base_adasyn(x, y, c, rate=1, m_neighbors=5, n_neighbors=5):
    gen_x = []
    gen_y = []

    # generate synthetic data for class c
    global_nbrs = NearestNeighbors(n_neighbors=m_neighbors).fit(x)
    min_idxs = np.where(y == c)[0]
    maj_idxs = np.where(y != c)[0]

    # calculate the weights for each minority instance
    # the only difference from blmovgen
    _, min_nbrs_ids = global_nbrs.kneighbors(x[min_idxs])
    weights = np.array([(y[id] != c).sum() for id in min_nbrs_ids])
    if weights.sum() == 0:
        return x, y

    # find nearest majority neighbors for each minority instance
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x[min_idxs])
    _, indices = nbrs.kneighbors(x[min_idxs])

    # generate synthetic data
    for j in np.random.choice(len(min_idxs), size=weights.sum() * rate, p=weights / weights.sum()if weights.sum() != 0 else None):
        min_idx = min_idxs[j]
        rand_idx = np.random.randint(1, n_neighbors)
        fac = np.random.rand()
        min_idx1 = min_idxs[indices[j][rand_idx]]
        new_x = x[min_idx] * fac + x[min_idx1] * (1 - fac)
        gen_x.append(new_x)
        gen_y.append(c)
    gen_x = np.array(gen_x)
    gen_y = np.array(gen_y)
    return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)


# TODO: adboth class: adasyn for majority and movgen(blmovgen, admovgen) for minority
def adboth(x, y, m_neighbors=5, n_neighbors=5, alpha=1.0):
    # generate synthetic data for each majority class
    # TODO: different idea to generate for majority class: 
        # 1. use the same weights as minority class
        # 2. use the weights of majority class itself
    # solution 1
    # sample strategy for majority classes
    _, counts = np.unique(y, return_counts=True)
    sample_size = max(counts) - counts
    for i, size in enumerate(sample_size):
        if size == 0:
            x, y = base_adasyn(x, y, i, rate=1, m_neighbors=m_neighbors, n_neighbors=n_neighbors)
    
    
    # sample strategy for each minority classes
    _, counts = np.unique(y, return_counts=True)
    sample_size = max(counts) - counts

    gen_x = []
    gen_y = []

    # generate synthetic data for each minority class
    global_nbrs = NearestNeighbors(n_neighbors=m_neighbors).fit(x)
    for i, size in enumerate(sample_size):
        if size == 0:
            continue
        min_idxs = np.where(y == i)[0]
        maj_idxs = np.where(y != i)[0]

        # calculate the weights for each minority instance
        # the only difference from blmovgen
        _, min_nbrs_ids = global_nbrs.kneighbors(x[min_idxs])
        weights = np.array([(y[id] == i).sum() for id in min_nbrs_ids])

        # find nearest majority neighbors for each minority instance
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x[maj_idxs])
        _, indices = nbrs.kneighbors(x[min_idxs])

        # generate synthetic data
        for j in np.random.choice(len(min_idxs), size=size, p=weights / weights.sum()):
            min_idx = min_idxs[j]
            rand_idx = np.random.randint(1, n_neighbors)
            maj_idx1 = maj_idxs[indices[j][0]]
            maj_idx2 = maj_idxs[indices[j][rand_idx]]
            new_x = x[min_idx] + alpha * (x[maj_idx2] - x[maj_idx1])
            gen_x.append(new_x)
            gen_y.append(i)
    gen_x = np.array(gen_x)
    gen_y = np.array(gen_y)
    return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)

# (0.8154584169387817, 0.5454545617103577, 0.4285714328289032, 0.857988178730011 ) nonsampling
# (0.8957356214523315, 0.6804123520851135, 0.9428571462631226, 0.8165680766105652) adasyn
# (0.8432836532592773, 0.6034482717514038, 1.0,                0.7278106212615967) blmovgen
# (0.868869960308075,  0.6476190686225891, 0.9714285731315613, 0.7810651063919067) admovgen




# blmix class for binary classification
def blmix(x, y, n_neighbors=5, alpha=0.5, **kwargs):
    # sample strategy for each minority classes
    _, counts = np.unique(y, return_counts=True)
    sample_size = max(counts) - counts

    gen_x = []
    gen_y = []

    # generate synthetic data for each minority class
    for i, size in enumerate(sample_size):
        if size == 0:
            continue
        min_idxs = np.where(y == i)[0]
        maj_idxs = np.where(y != i)[0]

        # find nearest majority neighbors for each minority instance
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x[maj_idxs])
        _, indices = nbrs.kneighbors(x[min_idxs])

        # generate synthetic data
        for j in np.random.randint(0, len(min_idxs), size):
            min_idx = min_idxs[j]
            rand_idx = np.random.randint(n_neighbors)
            maj_idx2 = maj_idxs[indices[j][rand_idx]]
            new_x = x[min_idx] * alpha + x[maj_idx2] * (1 - alpha)
            gen_x.append(new_x)
            gen_y.append(i)
    gen_x = np.array(gen_x)
    gen_y = np.array(gen_y)
    return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)

# blmixrand class for binary classification
def blmixrand(x, y, n_neighbors=5, alpha=0.5, **kwargs):
    # sample strategy for each minority classes
    _, counts = np.unique(y, return_counts=True)
    sample_size = max(counts) - counts

    gen_x = []
    gen_y = []

    # generate synthetic data for each minority class
    for i, size in enumerate(sample_size):
        if size == 0:
            continue
        min_idxs = np.where(y == i)[0]
        maj_idxs = np.where(y != i)[0]

        # find nearest majority neighbors for each minority instance
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x[maj_idxs])
        _, indices = nbrs.kneighbors(x[min_idxs])

        # generate synthetic data
        for j in np.random.randint(0, len(min_idxs), size):
            min_idx = min_idxs[j]
            rand_idx = np.random.randint(n_neighbors)
            fac = np.random.uniform(alpha, 1)
            maj_idx2 = maj_idxs[indices[j][rand_idx]]
            new_x = x[min_idx] * alpha + x[maj_idx2] * (1 - alpha)
            gen_x.append(new_x)
            gen_y.append(i)
    gen_x = np.array(gen_x)
    gen_y = np.array(gen_y)
    return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)