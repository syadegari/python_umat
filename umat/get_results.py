from typing import Any, Iterator, Tuple
from jaxtyping import Float
import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from umat_ggcm.pyUMATLink import pysdvini, fortumat


@dataclass
class UMATResults:
    F_final: Float[ndarray, "3 3"] = None
    F: Float[ndarray, "..."] = None
    theta: Float[ndarray, "3"] = None
    stress: Float[ndarray, "... 6"] = None
    intvars: Float[ndarray, "... 93"] = None
    ddsdde: Float[ndarray, "... 6 6"] = None


def get_F(f_init: Float[ndarray, "3 3"], f_final: Float[ndarray, "3 3"], t: float) -> Float[ndarray, "3 3"]:
    assert 0.0 <= t <= 1.0
    return (1 - t) * f_init + t * f_final


def get_ts(n_time: int) -> Float[ndarray, "..."]:
    return np.linspace(0, 1, n_time + 1)


def pairwise(xs: list[Any]) -> Iterator[Tuple[Any, Any]]:
    return zip(xs[:-1], xs[1:])


def get_results_from_umat(f_final: Float[ndarray, "3 3"], angles: Float[ndarray, "3"], n_times: int) -> dict:

    sigma = np.zeros(6)
    c = np.zeros([6, 6], order="F")
    xi = np.zeros(93)  # For ferritic phase
    init_temperature = 300.0
    f_init = np.eye(3)

    pysdvini(xi)

    hist_stress = [np.zeros(6)]
    hist_intvars = [xi.copy()]
    hist_ddsdde = []
    hist_F = [np.eye(3).ravel()]

    ts = get_ts(n_times)

    for t1, t2 in pairwise(ts):
        F1 = get_F(f_init, f_final, t1)
        F2 = get_F(f_init, f_final, t2)

        fortumat(F1, F2, t1, t2, init_temperature, sigma, c, xi, angles)

        hist_stress.append(sigma.copy())
        hist_intvars.append(xi.copy())
        hist_ddsdde.append(c.copy())
        hist_F.append(F2.ravel().copy())

    return {
        "F_final": f_final,
        "F": np.array(hist_F),
        "theta": angles,
        "stress": np.array(hist_stress),
        "intvars": np.array(hist_intvars),
        "ddsdde": np.array(hist_ddsdde),
        "ts": ts,
    }


@dataclass
class UMATResult:
    F_final: Float[ndarray, "3 3"] = None
    theta: Float[ndarray, "3"] = None
    F: Float[ndarray, "n_times+1 9"] = None
    stress: Float[ndarray, "n_times+1 6"] = None
    intvars: Float[ndarray, "n_times+1  93"] = None
    ts: Float[ndarray, "n_times+1"] = None


def get_results_with_state(f_final: Float[ndarray, "3 3"], angles: Float[ndarray, "3"], n_times: int) -> UMATResult:
    result = get_results_from_umat(f_final, angles, n_times)
    return UMATResult(
        F_final=result["F_final"],
        theta=result["theta"],
        F=result["F"],
        stress=result["stress"],
        intvars=result["intvars"],
        ts=result["ts"],
    )
