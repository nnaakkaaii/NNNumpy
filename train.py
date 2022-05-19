if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    from nnnumpy import criterion, engine, nn, optim
    from nnnumpy.data import transforms

    X, Y = fetch_openml('mnist_784', version=1, data_home="./data/", return_X_y=True)
    X = np.array(X/255.0, dtype=np.float32)
    Y = np.array(Y, dtype=np.uint8)

    train_val_x, test_x, train_val_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=2)
    train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=0.2, random_state=2)
    train_val_y = np.eye(10)[train_val_y].astype(np.int32)
    train_y = np.eye(10)[train_y].astype(np.int32)
    val_y = np.eye(10)[val_y].astype(np.int32)
    test_y = np.eye(10)[test_y].astype(np.int32)

    net = nn.ModuleList([
        nn.Linear(784, 1024),
        nn.BatchNormalization(1, 0),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 1024),
        nn.BatchNormalization(1, 0),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 1024),
        nn.BatchNormalization(1, 0),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 10),
    ])

    engine.engine(
        net,
        criterion.BCEWithLogitsLoss(),
        optim.AdamOptimizer(net.parameters(), lr=0.1, beta1=0.8, beta2=0.99),
        transforms.RotateTransform(20, 0.2, (28, 28)),
        train_val_x,
        train_val_y,
        test_x,
        test_y,
        n_epoch=20,
        batch_size=512,
    )
