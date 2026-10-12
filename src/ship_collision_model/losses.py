"""Loss functions and distribution metrics for neural network models."""

import numpy as np
import tensorflow as tf
from scipy.special import gammaln
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import Loss


class GammaNLLLoss(Loss):
    """Negative log-likelihood loss for a one-dimensional gamma distribution."""

    def call(self, y_true, y_pred):
        k = tf.maximum(y_pred[:, 0], 1e-6)
        theta = tf.maximum(y_pred[:, 1], 1e-6)
        y_true = tf.maximum(y_true, 1e-6)
        nll = k * tf.math.log(theta) + tf.math.lgamma(k) - (k - 1) * tf.math.log(y_true) + y_true / theta
        return tf.reduce_mean(nll)


class LogNormalNLLLoss(Loss):
    """Negative log-likelihood loss for a one-dimensional log-normal distribution."""

    def call(self, y_true, y_pred):
        mu = y_pred[:, 0]
        raw_sigma = y_pred[:, 1]
        sigma = tf.nn.softplus(raw_sigma) + 1e-6
        y_true = tf.maximum(y_true, 1e-6)
        logy = tf.math.log(y_true)
        nll = (
            0.5 * tf.square((logy - mu) / sigma)
            + tf.math.log(sigma)
            + logy
            + 0.5 * np.log(2.0 * np.pi)
        )
        return tf.reduce_mean(nll)


class LogNormalMetrics1D:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    @staticmethod
    def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def logpdf(self, y, mu, sigma):
        y = np.maximum(y, self.eps)
        logy = np.log(y)
        return -0.5 * ((logy - mu) / sigma) ** 2 - np.log(sigma) - np.log(y) - 0.5 * np.log(2.0 * np.pi)

    def total_log_likelihood(self, y, mu, sigma):
        return np.sum(self.logpdf(y, mu, sigma))

    def mean_nll(self, y, mu, sigma):
        return -np.mean(self.logpdf(y, mu, sigma))

    @staticmethod
    def aic(log_likelihood, k_params):
        return 2 * k_params - 2 * log_likelihood

    @staticmethod
    def bic(log_likelihood, k_params, n):
        return k_params * np.log(n) - 2 * log_likelihood

    def from_predictions(self, y_true, pred_mu_rawsigma, k_params=None):
        y = np.asarray(y_true).reshape(-1)
        mu = pred_mu_rawsigma[:, 0].reshape(-1)
        raw_sigma = pred_mu_rawsigma[:, 1].reshape(-1)
        sigma = self.softplus(raw_sigma) + 1e-6

        total_log_likelihood = self.total_log_likelihood(y, mu, sigma)
        avg_log_likelihood = total_log_likelihood / y.shape[0]
        out = {
            "N": y.shape[0],
            "avg_logL": avg_log_likelihood,
            "mean_NLL": -avg_log_likelihood,
            "total_logL": total_log_likelihood,
        }
        if k_params is not None:
            out["AIC"] = self.aic(total_log_likelihood, k_params)
            out["BIC"] = self.bic(total_log_likelihood, k_params, y.shape[0])
        return out


class GammaMetrics1D:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    @staticmethod
    def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def logpdf(self, y, k, theta):
        y = np.maximum(y, self.eps)
        return (k - 1) * np.log(y) - (y / theta) - k * np.log(theta) - gammaln(k)

    def total_log_likelihood(self, y, k, theta):
        return np.sum(self.logpdf(y, k, theta))

    def mean_nll(self, y, k, theta):
        return -np.mean(self.logpdf(y, k, theta))

    @staticmethod
    def aic(log_likelihood, k_params):
        return 2 * k_params - 2 * log_likelihood

    @staticmethod
    def bic(log_likelihood, k_params, n):
        return k_params * np.log(n) - 2 * log_likelihood

    def from_predictions(self, y_true, pred_k_theta, k_params=None, apply_softplus=False):
        y = np.asarray(y_true).reshape(-1)
        raw_k = pred_k_theta[:, 0].reshape(-1)
        raw_theta = pred_k_theta[:, 1].reshape(-1)
        if apply_softplus:
            k = self.softplus(raw_k) + 1e-6
            theta = self.softplus(raw_theta) + 1e-6
        else:
            k = raw_k
            theta = raw_theta

        total_log_likelihood = self.total_log_likelihood(y, k, theta)
        avg_log_likelihood = total_log_likelihood / y.shape[0]
        out = {
            "N": y.shape[0],
            "avg_logL": avg_log_likelihood,
            "mean_NLL": -avg_log_likelihood,
            "total_logL": total_log_likelihood,
        }
        if k_params is not None:
            out["AIC"] = self.aic(total_log_likelihood, k_params)
            out["BIC"] = self.bic(total_log_likelihood, k_params, y.shape[0])
        return out


class NNProcess:
    @staticmethod
    def split_and_train(X, Y, model, name, epochs=20, batch_size=128):
        x_temp, x_test, y_temp, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)
        print(f"[{name}] Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=1,
        )
        test_results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_results.items()])
        print(f"[{name}] Test metrics -> {metrics_str}")
        return history, (x_train, y_train, x_val, y_val, x_test, y_test), test_results
