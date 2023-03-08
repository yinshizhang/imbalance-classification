import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from .data import load_data, resampling, test_split


# define the parameters
# dsname = 'vehicle'
# sampling = 'adasyn'
# # sampling = 'nonsampling'
# normalize = 'scale'
# seed = 0
# test_size = 0.2
# epochs = 1000

def svm_metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    gmean = np.sqrt(recall * precision)
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return acc, recall, precision, f1, gmean, auc, ap

def run(dsname, sampling, seed, test_size=0.5, outexten='', m_neighbors=10, n_neighbors=5):

    # load data
    x, y = load_data(dsname)
    x_train, x_test, y_train, y_test = test_split(
        x, y, test_size=test_size, seed=seed)
    x_train, y_train = resampling(
        x_train, y_train, sampling=sampling, m_neighbors=m_neighbors, n_neighbors=n_neighbors)

    # define the model
    model = SVC(kernel='rbf', gamma='scale', probability=True)

    # create log file
    fname = f"results/svm_results{'' if outexten == '' else '_' + outexten}.csv"
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write(
                "dataset,sampling,m_neighbors,n_neighbors,seed,acc,recall,precision,f1,gmean,auc,ap\n")
    log_header = f"{dsname},{sampling},{m_neighbors},{n_neighbors},{seed}"

    print(f"Start training {dsname} with {sampling} methods")
    # train the model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    # test the model
    score = svm_metric(y_test, y_pred)
    print("acc: {:.3f}, recall: {:.3f}, precision: {:.3f}, f1: {:.3f}, gmean: {:.3f}, auc: {:.3f}, ap: {:.3f}".format(*score))
    with open(fname, 'a') as f:
        f.write(f"{log_header},{','.join([str(s) for s in score])}\n")
    # save the model
    # torch.save(model.state_dict(),
    #            f'./models/{dsname}_{sampling}_{normalize}_{seed}.pth')


if __name__ == '__main__':
    # dss = ['vehicle', 'diabete', 'vowel', 'ionosphere', 'abalone']
    # algs = ['nonsampling', 'smote', 'blsmote',
    #         'adasyn', 'blmovgen', 'admovgen', 'adboth']
    # for ds in dss:
    #     for alg in algs:

    run('vehicle', 'adasyn')