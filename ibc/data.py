import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .samplers import blmovgen, admovgen, adboth, blmix, blmixrand, admix, blpar
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE


D_PATH = {'root': '/bask/homes/c/cxx075/Chenguang/datasets/ADASYN/',
          'vehicle': '0_dataset_54_vehicle_van_vs_other.csv',
          'diabete': '1_Diabetes_no_header.csv',
          'vowel': '2_vowel_hid_vs_other.csv',
          'ionosphere': '3_ionosphere_no_header.csv',
          'abalone': '4_Abalone_no_header.csv',
          'satimage': '5_dataset_186_satimage_4_vs_other.csv',
          'haberman': '6_dataset_43_haberman.csv'
          }


SAMPLERS = {'oversampling': RandomOverSampler,
            'smote': SMOTE,
            'adasyn': ADASYN,
            'blsmote': BorderlineSMOTE
            }


CUSTOM_SAMPLERS = {'blmovgen': blmovgen,
                   'admovgen': admovgen,
                   'adboth': adboth,
                   'blmix': blmix,
                   'blmixrand': blmixrand,
                   'admix': admix,
                   'blpar': blpar
                   }


# load data file
def load_data(ds_name):
    '''
    ds_name: str, name of dataset
    '''
    data = pd.read_csv(
        D_PATH['root'] + D_PATH[ds_name], header=None).to_numpy(dtype='float32')
    return data[:, :-1], data[:, -1]


# train test split
def test_split(x, y, test_size=0.2, normalize='scale', seed=0):
    '''
    x: numpy array
    y: numpy array
    test_size: float, proportion of test set
    seed: int, random seed
    '''
    np.random.seed(seed)
    pidx = np.where(y == 1)[0]
    nidx = np.where(y == 0)[0]
    np.random.shuffle(pidx)
    np.random.shuffle(nidx)
    split = int(len(pidx) * test_size), int(len(nidx) * test_size)
    testid = np.concatenate((pidx[:split[0]], nidx[:split[1]]))
    trainid = np.concatenate((pidx[split[0]:], nidx[split[1]:]))

    x_test, x_train = x[testid], x[trainid]
    y_test, y_train = y[testid], y[trainid]
    # normalize the data into same scale
    if normalize == 'scale':
        x_train /= x_train.max(0)  # normalize is vital for MLP
        x_test /= x_test.max(0)  # normalize is vital for MLP
    elif normalize == 'standard':
        x_train = (x_train - x_train.mean(0)) / x.std(0)  # standardize
        x_test = (x_test - x_test.mean(0)) / x.std(0)  # standardize
    return x_train, x_test, y_train, y_test


# do resampling early than convert to dataset.
def resampling(x, y, sampling='nonsampling', **kwargs):
    '''
    x: numpy array
    y: numpy array
    sampling: str, 'nonsampling', 'oversampling', 'smote', 'adasyn', 'blsmote', 'adsmote', 'adboth'
    seed: int, random seed
    **kwargs: m_neighbors, n_neighbors, alpha
    '''

    if sampling == 'nonsampling':
        return x, y
    elif sampling in SAMPLERS:
        ros = SAMPLERS[sampling]()
        x_resampled, y_resampled = ros.fit_resample(x, y)
    elif sampling in CUSTOM_SAMPLERS:
        x_resampled, y_resampled = CUSTOM_SAMPLERS[sampling](x, y, **kwargs)
    else:
        raise ValueError(f'Unknown sampling method: {sampling}.')
    # TODO: implement custom resampling methods

    return x_resampled, y_resampled


class csvDS(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
