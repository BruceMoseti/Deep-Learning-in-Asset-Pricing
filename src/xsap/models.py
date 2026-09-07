"""Forecasting models, ordered by complexity.

Every model exposes the same two-line interface::

    model.fit(x_train, y_train, x_valid, y_valid)
    model.predict(x)

Hyper-parameters are chosen inside ``fit`` using the validation block only, so
the walk-forward harness never has to know anything model-specific and no
model can accidentally see the test block.  Selection is by validation mean
squared error on the cross-sectionally standardised target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from xsap.config import SEED

Array = np.ndarray


def _mse(y_true: Array, y_pred: Array) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


@dataclass
class SingleFeature:
    """Rank of one predictor, used unaltered as the forecast.

    The point of comparison for everything else: if a fitted model cannot beat
    a single standardised characteristic, its extra machinery earns nothing.
    """

    feature: str = "mom_2_12"
    name: str = "single-signal"
    _column: int = 0

    def fit(self, x_train, y_train, x_valid, y_valid, feature_names=None):
        assert feature_names is not None, "SingleFeature needs feature names"
        self._column = list(feature_names).index(self.feature)
        # Sign is set on the training block only, so the baseline is honest
        # about not knowing the direction of the effect in advance.
        signal = x_train[:, self._column]
        self._sign = float(np.sign(np.corrcoef(signal, y_train)[0, 1]) or 1.0)
        return self

    def predict(self, x):
        return self._sign * x[:, self._column]


@dataclass
class Linear:
    """OLS, or a penalised linear model with the penalty tuned on validation."""

    kind: str = "ols"
    # Weighted toward small penalties: the predictors are rank-transformed to
    # [-1, 1] and the target has unit cross-sectional variance, so a Lasso
    # penalty of 0.1 already zeroes every coefficient.  The grid needs
    # resolution below that, or the only choices on offer are "barely
    # penalised" and "no forecast at all".
    alphas: tuple[float, ...] = (
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0,
    )
    l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9)
    name: str = ""
    best: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = self.name or self.kind

    def _candidates(self):
        if self.kind == "ols":
            return [({}, LinearRegression())]
        if self.kind == "ridge":
            return [({"alpha": a}, Ridge(alpha=a)) for a in self.alphas]
        if self.kind == "lasso":
            return [
                ({"alpha": a}, Lasso(alpha=a, max_iter=20000)) for a in self.alphas
            ]
        if self.kind == "enet":
            return [
                (
                    {"alpha": a, "l1_ratio": r},
                    ElasticNet(alpha=a, l1_ratio=r, max_iter=20000),
                )
                for a in self.alphas
                for r in self.l1_ratios
            ]
        raise ValueError(f"unknown linear kind: {self.kind}")

    def fit(self, x_train, y_train, x_valid, y_valid, feature_names=None):
        best_score, best = np.inf, None
        for params, estimator in self._candidates():
            estimator.fit(x_train, y_train)
            score = _mse(y_valid, estimator.predict(x_valid))
            if score < best_score:
                best_score, best, self.best = score, estimator, params
        self._estimator = best
        return self

    def predict(self, x):
        return self._estimator.predict(x)

    @property
    def coef_(self) -> Array:
        return np.asarray(self._estimator.coef_).ravel()


@dataclass
class GradientBoosted:
    """XGBoost with depth and shrinkage tuned on validation, plus early stopping."""

    depths: tuple[int, ...] = (2, 3, 4)
    learning_rates: tuple[float, ...] = (0.03, 0.1)
    n_estimators: int = 800
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    seed: int = SEED
    name: str = "xgboost"
    best: dict = field(default_factory=dict)

    def fit(self, x_train, y_train, x_valid, y_valid, feature_names=None):
        import xgboost as xgb

        best_score, best = np.inf, None
        for depth in self.depths:
            for eta in self.learning_rates:
                model = xgb.XGBRegressor(
                    max_depth=depth,
                    learning_rate=eta,
                    n_estimators=self.n_estimators,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    reg_lambda=1.0,
                    min_child_weight=10,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=self.seed,
                    n_jobs=8,
                    early_stopping_rounds=30,
                )
                model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
                score = _mse(y_valid, model.predict(x_valid))
                if score < best_score:
                    best_score, best = score, model
                    self.best = {
                        "max_depth": depth,
                        "learning_rate": eta,
                        "n_estimators": int(model.best_iteration) + 1,
                    }
        self._model = best
        return self

    def predict(self, x):
        return self._model.predict(x)

    @property
    def feature_importance(self) -> Array:
        return self._model.feature_importances_


@dataclass
class NeuralNet:
    """Feed-forward network, ensembled over random initialisations.

    Architecture and weight decay are chosen on validation using one seed; the
    winning configuration is then refit under several seeds and the forecasts
    averaged.  Averaging over seeds matters: a single network's out-of-sample
    forecast on data this noisy is dominated by initialisation variance.
    """

    architectures: tuple[tuple[int, ...], ...] = ((32,), (32, 16), (32, 16, 8))
    weight_decays: tuple[float, ...] = (1e-5, 1e-3)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 4096
    max_epochs: int = 100
    patience: int = 8
    n_ensemble: int = 3
    seed: int = SEED
    name: str = "neural-net"
    best: dict = field(default_factory=dict)

    def _train_one(self, x_train, y_train, x_valid, y_valid, hidden, decay, seed):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        torch.set_num_threads(8)

        layers, in_dim = [], x_train.shape[1]
        for width in hidden:
            layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        net = nn.Sequential(*layers)

        optim = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=decay
        )
        loss_fn = nn.MSELoss()

        xt = torch.tensor(x_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        xv = torch.tensor(x_valid, dtype=torch.float32)
        yv = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

        generator = torch.Generator().manual_seed(seed)
        best_loss, best_state, stale = np.inf, None, 0
        for _ in range(self.max_epochs):
            net.train()
            order = torch.randperm(len(xt), generator=generator)
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                optim.zero_grad()
                loss_fn(net(xt[idx]), yt[idx]).backward()
                optim.step()
            net.eval()
            with torch.no_grad():
                loss = float(loss_fn(net(xv), yv))
            if loss < best_loss - 1e-7:
                best_loss, stale = loss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net, best_loss

    def fit(self, x_train, y_train, x_valid, y_valid, feature_names=None):
        x_train = np.ascontiguousarray(x_train, dtype=np.float32)
        x_valid = np.ascontiguousarray(x_valid, dtype=np.float32)
        y_train = np.ascontiguousarray(y_train, dtype=np.float32)
        y_valid = np.ascontiguousarray(y_valid, dtype=np.float32)

        best_loss, chosen = np.inf, None
        for hidden in self.architectures:
            for decay in self.weight_decays:
                _, loss = self._train_one(
                    x_train, y_train, x_valid, y_valid, hidden, decay, self.seed
                )
                if loss < best_loss:
                    best_loss, chosen = loss, (hidden, decay)
        hidden, decay = chosen
        self.best = {"hidden": hidden, "weight_decay": decay}

        self._nets = [
            self._train_one(
                x_train, y_train, x_valid, y_valid, hidden, decay, self.seed + k
            )[0]
            for k in range(self.n_ensemble)
        ]
        return self

    def predict(self, x):
        import torch

        tensor = torch.tensor(np.ascontiguousarray(x, dtype=np.float32))
        with torch.no_grad():
            preds = [net(tensor).squeeze(1).numpy() for net in self._nets]
        return np.mean(preds, axis=0)


def default_models() -> list:
    """The model ladder, in increasing order of flexibility."""
    return [
        SingleFeature(),
        Linear(kind="ols"),
        Linear(kind="ridge"),
        Linear(kind="lasso"),
        Linear(kind="enet"),
        GradientBoosted(),
        NeuralNet(),
    ]
