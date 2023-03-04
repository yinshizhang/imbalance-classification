import os
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from .models import MLP
from .metrics import Metrics
from .data import load_data, resampling, test_split, csvDS


# define the parameters
# dsname = 'vehicle'
# sampling = 'adasyn'
# # sampling = 'nonsampling'
# normalize = 'scale'
# seed = 0
# test_size = 0.2
# epochs = 1000

def run(dsname, sampling, seed, test_size, epochs, outexten=''):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    x, y = load_data(dsname)
    x_train, x_test, y_train, y_test = test_split(
        x, y, test_size=test_size, seed=seed)
    x_train, y_train = resampling(
        x_train, y_train, sampling=sampling)
    # convert to dataset
    train_ds = csvDS(x_train, y_train)
    test_ds = csvDS(x_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

    # define the model
    model = MLP(input_size=x_train.shape[1]).to(device)

    # define the loss function
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    evaluator = Metrics(device)

    # create log file
    fname = f"results{'' if outexten == '' else '_' + outexten}.csv"
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write(
                "dataset,sampling,seed,epoch,acc,recall,precision,f1,gmean,auc,ap\n")
    log_header = f"{dsname},{sampling},{seed},"

    print(f"Start training {dsname} with {sampling} methods")
    # train the model
    for t in range(epochs):
        # train
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f'Epoch {t}, train loss: {loss.item():.4f}')

        # test
        model.eval()
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                # acc, recall, precision, f1, gmean, auc, ap
                score = evaluator(y_pred, y)
        print("Epoch {} acc: {:.3f}, recall: {:.3f}, precision: {:.3f}, f1: {:.3f}, gmean: {:.3f}, auc: {:.3f}, ap: {:.3f}".format(t, *score))
        with open(fname, 'a') as f:
            f.write(f"{log_header}{t},{','.join([str(s) for s in score])}\n")
    # save the model
    # torch.save(model.state_dict(),
    #            f'./models/{dsname}_{sampling}_{normalize}_{seed}.pth')


if __name__ == '__main__':
    # dss = ['vehicle', 'diabete', 'vowel', 'ionosphere', 'abalone']
    # algs = ['nonsampling', 'smote', 'blsmote',
    #         'adasyn', 'blmovgen', 'admovgen', 'adboth']
    # for ds in dss:
    #     for alg in algs:

    run('vehicle', 'adasyn', 0, 0.2, 10000, 'adasyn')