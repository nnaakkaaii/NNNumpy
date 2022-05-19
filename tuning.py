import datetime
import json
import os
from typing import Dict, List, Type

import optuna
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from nnnumpy import base, criterion, nn, optim
from nnnumpy.data import transforms
from nnnumpy.engine import engine


class Tuner:

    acts: Dict[str, Type[base.Module]] = {
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'relu': nn.ReLU,
    }
    trans: Dict[str, base.Transform] = {
        'no_transform': transforms.NoTransform(),
        'rotate_transform': transforms.RotateTransform(20, 0.2, (28, 28)),
    }

    def __init__(self,
                 result_dir: str = './tuning_results',
                 n_epoch: int = 100,
                 batch_size: int = 2048) -> None:
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir
        self.n_epochs = n_epoch
        self.batch_size = batch_size

        t_delta = datetime.timedelta(hours=9)
        self.jst = datetime.timezone(t_delta, 'JST')

        # データセット
        X, Y = fetch_openml('mnist_784', version=1, data_home="./data/", return_X_y=True)
        X = np.array(X/255.0, dtype=np.float32)
        Y = np.array(Y, dtype=np.uint8)

        train_val_x, _, train_val_y, _ = train_test_split(X, Y, test_size=0.2, random_state=2)
        self.train_x, self.val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=0.2, random_state=2)
        self.train_y = np.eye(10)[train_y].astype(np.int32)
        self.val_y = np.eye(10)[val_y].astype(np.int32)

    def optimize(self):
        study = optuna.create_study(study_name='study_0519',
                                    storage=f'sqlite:///{self.result_dir}/optuna_study_0519.db',
                                    load_if_exists=True,
                                    direction='maximize')
        study.optimize(self.objective, n_trials=100)
        return

    def objective(self, trial):
        # MLP
        #   layers
        n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 4)
        n_hidden_dims = trial.suggest_categorical('n_hidden_dims', [128, 256, 512, 1024, 2048])
        #   Dropout
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.6)
        #   Batch Norm
        use_batch_norm = trial.suggest_int('use_batch_norm', 0, 1)
        #   Activation
        act_name = trial.suggest_categorical('act_name', list(self.acts.keys()))

        # Optimizer
        #   lr
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-2)
        #   beta
        beta1 = trial.suggest_uniform('beta1', 0.500, 0.999)
        beta2 = trial.suggest_uniform('beta2', 0.900, 0.999)

        # Transform
        transform_name = trial.suggest_categorical('transform_name', list(self.trans.keys()))

        # Builder Network
        layers: List[base.Module] = []
        # 1層目
        layers += [nn.Linear(784, n_hidden_dims)]
        if bool(use_batch_norm):
            layers += [nn.BatchNormalization(1, 0)]
        layers += [self.acts[act_name]()]
        layers += [nn.Dropout(dropout_rate)]
        # 2-n層目
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(n_hidden_dims, n_hidden_dims)]
            if bool(use_batch_norm):
                layers += [nn.BatchNormalization(1, 0)]
            layers += [self.acts[act_name]()]
            layers += [nn.Dropout(dropout_rate)]
        # n+1層目
        layers += [nn.Linear(n_hidden_dims, 10)]

        net = nn.ModuleList(layers)
        loss = criterion.BCEWithLogitsLoss()
        optimizer = optim.AdamOptimizer(net.parameters(), learning_rate, beta1, beta2)
        transform = self.trans[transform_name]

        # 実行
        history = engine(net,
                         loss,
                         optimizer,
                         transform,
                         self.train_x.copy(),
                         self.train_y.copy(),
                         self.val_x.copy(),
                         self.val_y.copy(),
                         n_epoch=self.n_epochs,
                         batch_size=self.batch_size)

        # 記録
        max_acc_score = 0
        max_acc_iter = 0
        dt = datetime.datetime.now(self.jst).strftime("%m%d_%H%M")
        for i, v in enumerate(history['test_acc']):
            if v > max_acc_score:
                max_acc_score = v
                max_acc_iter = i
        record = {
            'max_acc_score': max_acc_score,
            'max_acc_iter': max_acc_iter,
            'dt': dt,
            'n_hidden_layers': n_hidden_layers,
            'n_hidden_dims': n_hidden_dims,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm,
            'act_name': act_name,
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2,
            'transform_name': transform_name,
            'history': history,
        }
        with open(os.path.join(self.result_dir, f'{dt}_{max_acc_score}_at_{max_acc_iter}.json'), 'wb') as f:
            json.dump(record, f)

        return max_acc_score


if __name__ == '__main__':
    tuner = Tuner()
    tuner.optimize()
