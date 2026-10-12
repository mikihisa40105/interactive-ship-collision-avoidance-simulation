def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-((x - mu)**2) / (2 * sigma**2))


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, norm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from .losses import LogNormalNLLLoss


def mixture_gaussian_pdf(x, mu1, mu2, sigma1, sigma2, pi1, pi2):
    if not np.isclose(pi1 + pi2, 1.0):
        raise ValueError("Mixing coefficients pi1 and pi2 must sum to 1.")
    pdf1 = norm.pdf(x, loc=mu1, scale=sigma1)
    pdf2 = norm.pdf(x, loc=mu2, scale=sigma2)
    mixture_pdf = pi1 * pdf1 + pi2 * pdf2
    return mixture_pdf


def sample_from_mixture_gaussian(mu1, mu2, sigma1, sigma2, pi1, pi2):
    if type(mu1) == np.ndarray:
        mu1 = mu1.reshape(-1)[0]
        mu2 = mu2.reshape(-1)[0]
        sigma1 = sigma1.reshape(-1)[0]
        sigma2 = sigma2.reshape(-1)[0]
        pi1 = pi1.reshape(-1)[0]
        pi2 = pi2.reshape(-1)[0]

    probabilities = np.array([pi1, pi2])  # Convert to NumPy array
    probabilities /= probabilities.sum()   # Normalize probabilities to sum to 1
    np.random.choice([1, 2], p=probabilities)
    if not np.isclose(pi1 + pi2, 1.0):
        raise ValueError("Mixing coefficients pi1 and pi2 must sum to 1.")
    component = np.random.choice([1, 2], p=probabilities)
    if component == 1:
        sample = np.random.normal(loc=mu1, scale=sigma1)
    else:
        sample = np.random.normal(loc=mu2, scale=sigma2)

    return sample


class ModelIntegrator:
    """Integrate trained classifier and log-normal passage-position models.

    The repository includes Classification.h5, LogNormStb.h5, and
    LogNormPrt.h5 under models/ by default. Pass a fitted scaler or provide
    input_data.npy in model_dir.
    """

    def __init__(
        self,
        x_range=(-1.5, 1.5),
        num_points=500,
        plot=True,
        model_dir="./models",
        scaler=None,
        input_data_path=None,
        DIR=None,
    ):
        if DIR is not None:
            model_dir = DIR
        self.model_dir = Path(model_dir)
        self.classifier = load_model(self.model_dir / "Classification.h5")
        # input   x_cp_now, y_cp_now, tgt_length, own_length, own_spd, tgt_cog
        # output  1:Starbord 100%,  0:Port 100%
        self.cp_stb_predictor = load_model(
            self.model_dir / "LogNormStb.h5",
            custom_objects={
                "LogNormalNLLLoss": LogNormalNLLLoss,
                "log_normal_nll_loss": LogNormalNLLLoss,
            },
            compile=False,
        )
        # input   x_cp_now, y_cp_now, tgt_length, own_length, own_spd, tgt_cog
        # output  x_cp_future (mu, log_sigma)
        self.cp_prt_predictor = load_model(
            self.model_dir / "LogNormPrt.h5",
            custom_objects={
                "LogNormalNLLLoss": LogNormalNLLLoss,
                "log_normal_nll_loss": LogNormalNLLLoss,
            },
            compile=False,
        )
        # input   x_cp_now, y_cp_now, tgt_length, own_length, own_spd, tgt_cog
        # output  x_cp_future (mu, log_sigma)
        self.scaler = scaler or self._fit_scaler(input_data_path)
        self.x_range = x_range
        self.num_points = num_points
        self.x = np.linspace(self.x_range[0], self.x_range[1], self.num_points)
        self.combined_pdf = None
        if plot:
            plt.figure(figsize=(4.5, 3))
            plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
            plt.xlabel("Future CP-X [nm]", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.grid()

    def _fit_scaler(self, input_data_path):
        path = Path(input_data_path) if input_data_path else self.model_dir / "input_data.npy"
        if not path.exists():
            raise FileNotFoundError(
                "A fitted scaler or input_data.npy is required. "
                "Pass scaler=..., input_data_path=..., or place input_data.npy "
                f"under {self.model_dir}."
            )
        input_data = np.load(path).reshape(-1, 6)
        return StandardScaler().fit(input_data)

    def cal_pdf(self, mu_prt, sigma_prt, mu_stb, sigma_stb, pi_stb):
        pdf_stb = lognorm.pdf(self.x, s=sigma_stb, scale=np.exp(mu_stb))
        pdf_prt = lognorm.pdf(-self.x, s=sigma_prt, scale=np.exp(mu_prt))

        pi_prt = 1 - pi_stb
        self.combined_pdf = pi_stb * pdf_stb + pi_prt * pdf_prt
        
    def cal_cp_future_distribution(self, x_cp_now, y_cp_now, tgt_length, own_length, own_spd, tgt_cog):
        input_data = np.array([x_cp_now, y_cp_now, tgt_length, own_length, own_spd, tgt_cog])
        input_data = input_data.reshape(-1, 6)
        input_data = self.scaler.transform(input_data)
        
        pi_stb = self.classifier.predict(input_data, verbose=0)[0,0]
        
        cp_stb = self.cp_stb_predictor.predict(input_data, verbose=0)
        mu_stb = cp_stb[0][0]
        sigma_stb = np.log1p(np.exp(cp_stb[0][1])) + 1e-6

        cp_prt = self.cp_prt_predictor.predict(input_data, verbose=0)
        mu_prt = cp_prt[0][0]
        sigma_prt = np.log1p(np.exp(cp_prt[0][1])) + 1e-6
        
        #print(mu_prt, sigma_prt, mu_stb, sigma_stb, pi_stb)
        self.cal_pdf(mu_prt, sigma_prt, mu_stb, sigma_stb, pi_stb)
        return mu_prt, sigma_prt, mu_stb, sigma_stb, pi_stb
        
    def sampling_x_cp_future(self, x_cp_now, y_cp_now, tgt_length, own_length, own_spd, tgt_cog):
        mu_prt, sigma_prt, mu_stb, sigma_stb, pi_stb = self.cal_cp_future_distribution(
            x_cp_now, y_cp_now, tgt_length, own_length, own_spd, tgt_cog
        )
        if np.random.uniform(0.0, 1.0) < pi_stb:
            rng = np.random.default_rng() 
            x = rng.lognormal(mean=mu_stb, sigma=sigma_stb, size=1)
            return x[0]
        else:
            rng = np.random.default_rng() 
            x = rng.lognormal(mean=mu_prt, sigma=sigma_prt, size=1)
            return - x[0]

    def plot(self, label="type"):
        plt.plot(self.x, self.combined_pdf, label=label)

    def save_fig(self, filename="CombinedPdf.png"):
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)



