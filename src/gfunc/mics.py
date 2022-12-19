import warnings

import numpy as np
import pandas as pd
import torch

try:
    # new stdlib in Python3.9
    from importlib.resources import as_file, files
except ImportError:
    # third-party package, backport for Python3.9-,
    # need to add importlib_resources to requirements
    from importlib_resources import as_file, files

# MICS-ANN PYTORCH MODEL
with as_file(files("gfunc").joinpath("torchmodel/traced_micsann.pt")) as micspath:
    model = torch.jit.load(micspath).to("cpu").eval()


def mics(Fo, Pe, phi, R):
    # make input dataframe
    df = pd.DataFrame(
        {
            "Fo_rc": Fo,
            "Pe_rc": Pe * np.ones_like(Fo),
            "phi": phi * np.ones_like(Fo),
            "R": R * np.ones_like(Fo),
        }
    )
    # calc dimensionless temperature of MICS-ANN model
    with torch.no_grad():
        data = normalize(df).to("cpu")

        predicted_tensor = model(data)
        f_mics = predicted_tensor.to("cpu").detach().numpy().ravel().copy()
    return f_mics


def micsmean(Fo, Pe, R):
    Nmean = 10
    phi = np.linspace(0, np.pi, Nmean)
    # make input dataframe
    df = pd.DataFrame(
        {
            "Fo_rc": np.tile(Fo, Nmean),
            "Pe_rc": np.tile(np.full_like(Fo, Pe), Nmean),
            "phi": np.repeat(phi, len(Fo)),
            "R": np.tile(np.full_like(Fo, R), Nmean),
        }
    )
    # calc dimensionless temperature of MICS-ANN model
    with torch.no_grad():
        data = normalize(df).to("cpu")

        predicted_tensor = model(data)
        f_tmp = predicted_tensor.to("cpu").detach().numpy().ravel().copy()
    f_mics = np.mean(f_tmp.reshape((len(Fo), Nmean), order="F"), axis=1)
    return f_mics


# input normalization
def normalize(df):
    Fo_rc_min = 5e-4
    Fo_rc_max = 1e4
    Pe_rc_min = 1e-3
    Pe_rc_max = 1e-3 * 1.1 ** (100 - 1)
    phi_min = 0
    phi_max = np.pi
    R_min = 0.999871
    R_max = 6.0

    # throw warning when input is out of validation range
    if (df["Fo_rc"].min() < Fo_rc_min) or (Fo_rc_max < df["Fo_rc"].max()):
        warnings.warn(
            "Fo_rc is out of validation range ({0:.2e} < Fo_rc < {1:.2e}).".format(
                Fo_rc_min, Fo_rc_max
            ),
            stacklevel=3,
        )
    if (df["Pe_rc"].min() < Pe_rc_min) or (Pe_rc_max < df["Pe_rc"].max()):
        warnings.warn(
            "Pe_rc is out of validation range ({0:.2e} < Pe_rc < {1:.2e}).".format(
                Pe_rc_min, Pe_rc_max
            ),
            stacklevel=3,
        )
    if (df["phi"].min() < phi_min) or (phi_max < df["phi"].max()):
        warnings.warn(
            "phi is out of validation range ({0:.2f} < phi < {1:.2f}).".format(
                phi_min, phi_max
            ),
            stacklevel=3,
        )
    if (df["R"].min() < R_min) or (R_max < df["R"].max()):
        warnings.warn(
            "R is out of validation range ({0:.2f} < R < {1:.2f}).".format(
                R_min, R_max
            ),
            stacklevel=3,
        )

    df_norm = df.rename(columns={"Fo_rc": "log_Fo_rc", "Pe_rc": "log_Pe_rc"})
    df_norm["log_Fo_rc"] = (np.log(df["Fo_rc"]) - np.log(Fo_rc_min)) / (
        np.log(Fo_rc_max) - np.log(Fo_rc_min)
    )
    df_norm["log_Pe_rc"] = (np.log(df["Pe_rc"]) - np.log(Pe_rc_min)) / (
        np.log(Pe_rc_max) - np.log(Pe_rc_min)
    )
    df_norm["phi"] = (df["phi"] - phi_min) / (phi_max - phi_min)
    df_norm["R"] = (df["R"] - R_min) / (R_max - R_min)

    data = torch.tensor(df_norm[["log_Fo_rc", "log_Pe_rc", "phi", "R"]].values).float()

    return data
