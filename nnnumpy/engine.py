from typing import Dict, List

import numpy as np

from .base import Loss, Module, Optimizer, Transform


def engine(net: Module,
           criterion: Loss,
           optimizer: Optimizer,
           transform: Transform,
           train_x: np.ndarray,
           train_y: np.ndarray,
           test_x: np.ndarray,
           test_y: np.ndarray,
           n_epoch: int,
           batch_size: int) -> Dict[str, List[float]]:

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    train_n = len(train_x)
    test_n = len(test_x)

    for epoch in range(n_epoch):
        print('epoch %d | ' % epoch, end='')

        # TRAIN
        sum_loss = 0
        pred_y = []
        perm = np.random.permutation(train_n)

        net.train()

        for i in range(0, train_n, batch_size):
            x = transform(train_x[perm[i:i+batch_size]])
            t = train_y[perm[i:i+batch_size]]

            y = net(x)
            loss = criterion(y, t)
            dout = criterion.backward()
            _ = net.backward(dout)
            optimizer.step()

            sum_loss += loss
            pred_y.extend(np.argmax(y, axis=1))

        history['train_loss'] = loss = sum_loss / train_n
        history['train_acc'] = accuracy = np.sum(np.eye(10)[pred_y] * train_y[perm]) / train_n

        print('Train loss %.3f, accuracy %.4f | ' % (loss, accuracy), end="")

        # TEST
        sum_loss = 0
        pred_y = []

        net.eval()

        for i in range(0, test_n, batch_size):
            x = test_x[i: i+batch_size]
            t = test_y[i: i+batch_size]

            y = net(x)
            loss = criterion(y, t)

            sum_loss += loss
            pred_y.extend(np.argmax(y, axis=1))

        history['test_loss'] = loss = sum_loss / test_n
        history['test_acc'] = accuracy = np.sum(np.eye(10)[pred_y] * test_y) / test_n
        print('Test loss %.3f, accuracy %.4f' % (loss, accuracy))

    return history
