import io
import time
import os
import json
import h5py
import dataclasses
from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
from numpy import ndarray
from numpy import sin, cos
from jaxtyping import Float
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

# module imports
from .get_results import get_results_from_umat

# We want to generate deformation gradient and orientation pairs for simulations
# We only want to save these pairs and not the entire history
#
"""
The following criteria should be satisfied

- Deformation gradient should be close to isochoric
- Deformation gradient should have between 7 to 10 percent strain
- Maximum experienced stress should be below 800 MPa

The last condition requires that we pair each deformation gradient with an orientation (since the response depends on the orientation).
For that reason, we save the results in pairs of (deformation_gradent, angel).
"""


@dataclass
class Config:
    eps_condition1: float = None
    eps_lower_bound_condition2: float = None
    eps_upper_bound_condition2: float = None
    scale_factor: float = None
    max_allowable_stress: float = None
    n_times: int = None
    num_samples: int = None
    file_name: str = None


def generate_single_orientation():
    return np.array(
        [
            2 * np.pi * np.random.uniform(low=0, high=1.0),
            np.arccos(2 * np.random.uniform(low=0, high=1.0) - 1),
            2 * np.pi * np.random.uniform(low=0, high=1.0),
        ]
    )


def gl_strain(F: ndarray) -> ndarray:
    return 0.5 * (F.T @ F - np.eye(3))


def constraint1(F: ndarray, eps: float) -> bool:
    """almost-isochoric condition"""
    return 1 - eps < np.linalg.det(F) < 1 + eps


def constraint2(F: ndarray, eps_lower_bound: float, eps_upper_bound: float) -> bool:
    """condition on maximum component of E"""
    E = gl_strain(F)
    return eps_lower_bound < np.abs(E).max() < eps_upper_bound


def generate_defGrad(
    sc_factor: float,
    eps_constraint1: float,
    eps_lower_bound_constraint2: float,
    eps_upper_bound_constraint2: float,
) -> ndarray:
    """deforamtion gradient from normal distribution"""
    while True:
        candidate_defgrad = np.eye(3) + sc_factor * np.random.normal(size=9).reshape(3, 3)
        if constraint1(candidate_defgrad, eps_constraint1) and constraint2(
            candidate_defgrad, eps_lower_bound_constraint2, eps_upper_bound_constraint2
        ):
            return candidate_defgrad


def generate_data(cfg: Config):
    # Each process gets its own seed
    np.random.seed(int(time.time()) + os.getpid())

    defgrad = generate_defGrad(
        cfg.scale_factor, cfg.eps_condition1, cfg.eps_lower_bound_condition2, cfg.eps_upper_bound_condition2
    )
    angles = generate_single_orientation()

    results = get_results_from_umat(defgrad, angles, cfg.n_times)
    max_stress = np.abs(results["stress"]).max()
    if max_stress < cfg.max_allowable_stress:
        return {"defgrad": defgrad, "angles": angles, "max_stress": max_stress}
    else:
        return {}


def aux(cfg: Config) -> list[dict]:
    nproc = mp.cpu_count()
    with mp.Pool(processes=nproc) as pool:
        result = pool.map(generate_data, [cfg for _ in range(nproc)])

    return result


def main(cfg: Config) -> None:

    saved_samples = []

    while len(saved_samples) < cfg.num_samples:
        found_samples = aux(cfg)
        # import pdb; pdb.set_trace()
        found_samples = [r for r in found_samples if r]
        for sample in found_samples:
            saved_samples.append(sample)
        print(f"Found {len(saved_samples):<{int(np.log10(cfg.num_samples)) + 1}d} samples.")

    if len(saved_samples) > cfg.num_samples:
        print(f"Found {len(saved_samples) - cfg.num_samples} samples more than what is specified in cfg.")
        print(f"Only saving the first {cfg.num_samples} samples.")

    save_results(saved_samples[: cfg.num_samples], cfg)


class EnhancedJSONEncoder(json.JSONEncoder):
    """https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def rotationMatrix(ang1, ang2, ang3):
    R = np.array(
        [
            [
                cos(ang1) * cos(ang2) * cos(ang3) - sin(ang1) * sin(ang3),
                -cos(ang3) * sin(ang1) - cos(ang1) * cos(ang2) * sin(ang3),
                cos(ang1) * sin(ang2),
            ],
            [
                cos(ang1) * sin(ang3) + cos(ang2) * cos(ang3) * sin(ang1),
                cos(ang1) * cos(ang3) - cos(ang2) * sin(ang1) * sin(ang3),
                sin(ang1) * sin(ang2),
            ],
            [-cos(ang3) * sin(ang2), sin(ang2) * sin(ang3), cos(ang2)],
        ]
    )
    return R


def drawProjection2(ang1, ang2, ang3, ax) -> None:
    R = rotationMatrix(ang1, ang2, ang3)
    # rows of rotation matrix are the coordinates of the rotated crystal in the
    # global coordinate system.
    # we need points which are in the northern hemisphere (Z > 0). check
    # for this and if Z < 0, replace it with its mirror.
    for i in range(3):
        if R[i][2] < 0.0:
            R[i] = -R[i]
    A = np.array([0.0, 0.0, -1.0])
    ##    color = [ 'r', 'b', 'g']
    color = ["k", "k", "k"]
    for i in range(3):
        X = R[i]
        t = 1.0 / (1.0 + X[2])
        P = t * X + (1.0 - t) * A
        ax.plot(P[0], P[1], color[i] + ".", ms=1)


def plot_pole_figure(angles: list[ndarray]) -> io.BytesIO:
    fig, ax = plt.subplots()

    for angle in angles:
        drawProjection2(*angle, ax)

    circle = plt.Circle((0, 0), 1, color="k", fill=False, lw=0.4)
    ax.add_patch(circle)
    ax.axis([-1.1, 1.1, -1.1, 1.1])
    ticks = [-1, -0.5, 0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.axis("equal")
    plt.tight_layout()

    return fig_to_buffer(fig)


def get_TSNE_2d(X: Float[ndarray, "n_samples n_features"]) -> Float[ndarray, "n_samples 2"]:
    return TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(X)


def plot_defgrad_tsne_projection(defgrads: list[Float[ndarray, "3 3"]]) -> io.BytesIO:
    tsne_projection = get_TSNE_2d(np.array([x.ravel() for x in defgrads]))

    fig, ax = plt.subplots()
    ax.plot(tsne_projection[:, 0], tsne_projection[:, 1], "o", ms=0.5)
    ax.set_aspect("equal", "box")
    plt.tight_layout()

    return fig_to_buffer(fig)


def fig_to_buffer(fig: Figure, format: str = "png", dpi: int = 200) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf


def save_results(results: dict, cfg: Config) -> None:
    length = len(results)

    with h5py.File(f"{cfg.file_name}.h5", "w") as hdf:
        pairs_group = hdf.create_group("data_pairs")
        for idx, data in enumerate(results):
            grp = pairs_group.create_group(f"pair_{idx:0{len(str(length))}}")
            grp.create_dataset("defgrad", data=data["defgrad"], dtype="float64")
            grp.create_dataset("angle", data=data["angles"], dtype="float64")
            grp.create_dataset("max_stress", data=data["max_stress"], dtype="float64")

        angles = [result["angles"] for result in results]
        defgrads = [result["defgrad"] for result in results]

        buf_pole_fig = plot_pole_figure(angles)
        hdf.create_dataset("pole_figure", data=np.void(buf_pole_fig.getvalue()))
        buf_pole_fig.close()

        buf_tsne_fig = plot_defgrad_tsne_projection(defgrads)
        hdf.create_dataset("tsne_defgrad", data=np.void(buf_tsne_fig.getvalue()))
        buf_tsne_fig.close()

        cfg_json = json.dumps(cfg, cls=EnhancedJSONEncoder)
        hdf.attrs["config"] = cfg_json


if __name__ == "__main__":
    cfg = Config(
        eps_condition1=0.00125,
        eps_lower_bound_condition2=0.09,
        eps_upper_bound_condition2=0.1,
        scale_factor=0.1,
        max_allowable_stress=0.8,
        n_times=150,
        num_samples=10000,
        file_name="data_pairs",
    )
    main(cfg)
