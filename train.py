from nnnumpy import nn

Mem = {
    'Dense2Layer': nn.ModuleList([
        nn.Linear(784, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10),
    ]),
    'ConvPoolDense': nn.ModuleList([
        nn.Reshape((1, 28, 28)),
        nn.Conv2D(1, 32, 5),  # 32, 24, 24
        nn.ReLU(),
        nn.MeanPool(2, stride=2),  # 32, 12, 12
        nn.Dropout(0.4),
        nn.Conv2D(32, 64, 5),  # 64, 8, 8
        nn.ReLU(),
        nn.MeanPool(2, stride=2),  # 64, 4, 4
        nn.Dropout(0.4),
        nn.Reshape((64 * 4 * 4,)),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 10),
    ]),
    'ConvDense': nn.ModuleList([
        nn.Reshape((1, 28, 28)),
        nn.Conv2D(1, 32, 4, 2, 1),  # 32, 14, 14
        nn.ReLU(),
        nn.Conv2D(32, 64, 4, 2, 1),  # 64, 7, 7
        nn.ReLU(),
        nn.Reshape((64 * 7 * 7,)),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ]),
    'ConvConvDense': nn.ModuleList([
        nn.Reshape((1, 28, 28)),
        nn.Conv2D(1, 32, 3),  # 32, 26, 26
        nn.ReLU(),
        nn.Conv2D(32, 32, 3),  # 32, 24, 24
        nn.ReLU(),
        nn.Conv2D(32, 32, 5, stride=2),  # 32, 12, 12
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Conv2D(32, 64, 3),  # 64, 10, 10
        nn.ReLU(),
        nn.Conv2D(64, 64, 3),  # 64, 8, 8
        nn.ReLU(),
        nn.Conv2D(64, 64, 5, stride=2),  # 64, 4, 4
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Reshape((64 * 4 * 4,)),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 10),
    ]),
    'ConvConv': nn.ModuleList([
        nn.Reshape((1, 28, 28)),
        nn.Conv2D(1, 32, 5),  # 32, 24, 24
        nn.ReLU(),
        nn.Conv2D(32, 32, 4, 2, 1),  # 32, 12, 12
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Conv2D(32, 64, 5),  # 64, 8, 8
        nn.ReLU(),
        nn.Conv2D(64, 64, 4, 2, 1),  # 64, 4, 4
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Conv2D(64, 10, 4),  # 10, 1, 1
        nn.Reshape((10,)),
    ]),
}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, choices=list(Mem.keys()))
    args = parser.parse_args()

    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    from nnnumpy import criterion, optim, engine

    X, Y = fetch_openml('mnist_784', version=1, data_home="./data/", return_X_y=True)
    X = np.array(X/255.0, dtype=np.float32)
    Y = np.array(Y, dtype=np.uint8)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=2)
    train_y = np.eye(10)[train_y].astype(np.int32)
    test_y = np.eye(10)[test_y].astype(np.int32)

    net = Mem[args.network]

    engine.engine(
        net,
        criterion.BCEWithLogitsLoss(),
        optim.SGDOptimizer(net.parameters(), lr=0.1),
        train_x,
        train_y,
        test_x,
        test_y,
        n_epoch=20,
        batch_size=512,
    )
