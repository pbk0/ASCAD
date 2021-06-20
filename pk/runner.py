import sys
sys.path.append("..")
import enum
import pathlib
import numpy as np
import typing as t
import plotly.graph_objects as go
import pickle
import plotly.express as px
import sys
import pandas as pd
import os

ROOT_DIR = pathlib.Path(__file__).parent / "results"


def rank(predictions, key, targets, ntraces, interval=10):
    ranktime = np.zeros(int(ntraces/interval))
    pred = np.zeros(256)

    idx = np.random.randint(predictions.shape[0], size=ntraces)

    for i, p in enumerate(idx):
        for k in range(predictions.shape[1]):
            pred[k] += predictions[p, targets[p, k]]

        if i % interval == 0:
            ranked = np.argsort(pred)[::-1]
            ranktime[int(i/interval)] = list(ranked).index(key)

    return ranktime


class Experiment(enum.Enum):

    ascad_v1_fk_0_mlp = enum.auto()
    ascad_v1_fk_50_mlp = enum.auto()
    ascad_v1_fk_100_mlp = enum.auto()
    ascad_v1_fk_0_cnn = enum.auto()
    ascad_v1_fk_50_cnn = enum.auto()
    ascad_v1_fk_100_cnn = enum.auto()

    @property
    def plot_dir(self) -> pathlib.Path:
        _ret = ROOT_DIR / "plots" / self.name
        if not _ret.exists():
            _ret.mkdir(parents=True)
        return _ret

    @property
    def rank_plot_until(self) -> int:
        if self.name.startswith("ascad"):
            return 1000
        else:
            raise Exception(f"Unsupported {self}")

    def get_database_file_name(self) -> str:

        _ascad_database = "../ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/"

        if self.name.find("_0_") != -1:
            _ascad_database += "ASCAD.h5"
        elif self.name.find("_50_") != -1:
            _ascad_database += "ASCAD_desync50.h5"
        elif self.name.find("_100_") != -1:
            _ascad_database += "ASCAD_desync100.h5"
        else:
            raise Exception(f"Unsupported {self}")

        return _ascad_database

    def get_test_params(self, experiment_id: int) -> t.Dict:

        return {
            "ascad_database": self.get_database_file_name(),
            "model_file": (self.store_dir(experiment_id) / "model.h5").resolve().as_posix(),
            "avg_rank_file": (self.store_dir(experiment_id) / "avg_rank.npy").resolve().as_posix(),
            "num_traces": 10000,
        }

    def get_train_params(self, experiment_id: int) -> t.Dict:
        if self.name.startswith("ascad"):
            if self.name.endswith("_mlp"):
                _network_type = "mlp"
                _batch_size = 100
                _epochs = 200
            elif self.name.endswith("_cnn"):
                _network_type = "cnn"
                _batch_size = 200
                _epochs = 75
            else:
                raise Exception(f"Unsupported {self}")

            _ret = {
                "ascad_database": self.get_database_file_name(),
                "training_model": (self.store_dir(experiment_id) / "model.h5").resolve().as_posix(),
                "history_file": (self.store_dir(experiment_id) / "history.pickle").resolve().as_posix(),
                "network_type": _network_type,
                "epochs": _epochs,
                "batch_size": _batch_size,
            }
        else:
            raise Exception(f"Unsupported {self}")

        # return
        return _ret

    def store_dir(self, experiment_id: int):
        _ret = ROOT_DIR / self.name / str(experiment_id)
        if not _ret.exists():
            _ret.mkdir(parents=True)
        return _ret

    def exists(self, experiment_id: int) -> bool:
        _store_dir = self.store_dir(experiment_id)
        _model_file = _store_dir / "model.h5"
        _history_file = _store_dir / "history.pickle"
        _avg_rank_file = _store_dir / "avg_rank.npy"
        return _model_file.is_file() and _history_file.is_file() and _avg_rank_file.is_file()

    # noinspection PyPep8Naming
    def train_and_rank(self, experiment_id: int):
        # files that will be saved
        _store_dir = self.store_dir(experiment_id)
        _model_file = _store_dir / "model.hdf5"
        _history_file = _store_dir / "history.pickle"
        _avg_rank_file = _store_dir / "avg_rank.npy"
        _train_params_file = _store_dir / "train_params.txt"
        _test_params_file = _store_dir / "test_params.txt"

        # if model file present and other files also exist that means something was run so return
        if self.exists(experiment_id):
            return
        else:
            if _model_file.is_file():
                _model_file.unlink()
            if _history_file.is_file():
                _history_file.unlink()
            if _avg_rank_file.is_file():
                _avg_rank_file.unlink()
            if _train_params_file.is_file():
                _train_params_file.unlink()
            if _test_params_file.is_file():
                _test_params_file.unlink()

        # make params files
        _train_params_file.touch(exist_ok=False)
        _train_params_file.write_text(str(self.get_train_params(experiment_id=experiment_id)))
        _test_params_file.touch(exist_ok=False)
        _test_params_file.write_text(str(self.get_test_params(experiment_id=experiment_id)))

        # call
        print("***************************************************************")
        print("********************** TRAINING *******************************")
        print("***************************************************************")
        os.system(
            f"C:/Python38/python ../ASCAD_train_models.py {_train_params_file.as_posix()}")

        print()
        print()
        print()
        print("***************************************************************")
        print("*********************** TESTING *******************************")
        print("***************************************************************")
        os.system(
            f"C:/Python38/python ../ASCAD_test_models.py {_test_params_file.as_posix()}")

    def results(self):

        # first lets see check if some results were not converges
        non_converged_experiments = []
        for experiment_id in range(100):
            # get store dir
            store_dir = self.store_dir(experiment_id)
            # skip is results not present
            if not self.exists(experiment_id):
                continue
            # noinspection PyTypeChecker
            avg_rank = np.load(store_dir / f"avg_rank.npy")
            traces_with_rank_0 = np.where(avg_rank <= 0)[0]
            if len(traces_with_rank_0) == 0:
                non_converged_experiments.append(experiment_id)

        # create figures
        ranks_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(
                    text=f"{self.name}: Average Rank")
            )
        )
        loss_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{self.name}: Train Loss")
            )
        )
        accuracy_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{self.name}: Validation Loss")
            )
        )

        # plot
        min_traces_with_rank_0_per_experiment = []
        for experiment_id in range(100):
            # get store dir
            store_dir = self.store_dir(experiment_id)
            # skip is results not present
            if not self.exists(experiment_id):
                continue
            avg_rank = np.load(store_dir / f"avg_rank.npy")
            with open((store_dir / f"history.pickle").as_posix(), 'rb') as file_pi:
                history = pickle.load(file_pi)
            loss = history['loss']
            accuracy = history['accuracy']
            traces_with_rank_0 = np.where(avg_rank <= 0)[0]
            if len(traces_with_rank_0) == 0:
                min_traces_with_rank_0_per_experiment.append(np.inf)
            else:
                min_traces_with_rank_0_per_experiment.append(traces_with_rank_0[0])

            # noinspection DuplicatedCode
            rank_plot_until = self.rank_plot_until
            ranks_fig.add_trace(
                go.Scatter(
                    x=[avg_rank[i][0] for i in range(0, rank_plot_until)],
                    y=[avg_rank[i][1] for i in range(0, rank_plot_until)],
                    mode='lines',
                    name=f"exp_{experiment_id:03d}"
                )
            )
            loss_fig.add_trace(
                go.Scatter(
                    x=np.arange(len(loss)),
                    y=loss,
                    mode='lines',
                    name=f"exp_{experiment_id:03d}"
                )
            )
            accuracy_fig.add_trace(
                go.Scatter(
                    x=np.arange(len(accuracy)),
                    y=accuracy,
                    mode='lines',
                    name=f"exp_{experiment_id:03d}"
                )
            )

        # plots
        print("************************************************************************")
        print(f"> Results for {self.name}")
        if bool(non_converged_experiments):
            print(
                f"Note that {len(non_converged_experiments)} experiments did not converge"
            )
            print(non_converged_experiments)
        print("************************************************************************")
        print("plotting histogram")
        df = pd.DataFrame()
        df['min_traces_for_rank_0'] = [
            _ for _ in min_traces_with_rank_0_per_experiment if _ != np.inf
        ]
        hist_fig = px.histogram(
            df, x="min_traces_for_rank_0",
            title=f"{self.name}: Histogram",
        )
        hist_fig.write_image((self.plot_dir / f"hist.svg").as_posix())
        print("plotting ranks")
        ranks_fig.write_image((self.plot_dir / f"rank.svg").as_posix())
        print("plotting train loss")
        loss_fig.write_image((self.plot_dir / f"loss.svg").as_posix())
        print("plotting validation loss")
        accuracy_fig.write_image((self.plot_dir / f"accuracy.svg").as_posix())

        # write details to md file
        _md_file = self.plot_dir.parent / f"{self.name}.md"
        _md_file_lines = [
            f"# Results for {self.name}", "", f"## Convergence", ""
        ]
        if bool(non_converged_experiments):
            _md_file_lines += [
                f"**Note that `{len(non_converged_experiments)} %` experiments did not converge.**",
                f"Failed experiment id's are as below:",
                f"  > {non_converged_experiments}", ""
            ]
        else:
            _md_file_lines += [
                f"Note that all 100 experiments converged", ""
            ]
        _md_file_lines += [
            f"## Histogram",
            "",
            f"![Histogram - {self.name}]({self.name}/hist.svg)",
            "",
            f"## Ranks",
            "",
            f"![Rank - {self.name}]({self.name}/rank.svg)",
            "",
            # f"## Train Loss",
            # "",
            # f"![Train Loss - {self.name}]({self.name}/loss.svg)",
            # "",
            # f"## Validation Loss",
            # "",
            # f"![Validation Loss - {self.name}]({self.name}/accuracy.svg)",
            # "",
        ]
        _sorted_ids = np.argsort(min_traces_with_rank_0_per_experiment)
        _md_file_lines += [f"## Sorted Experiment IDs with min traces", "", "```"]
        for _id in _sorted_ids:
            _md_file_lines += [
                f"experiment {_id:03d} :: min traces {min_traces_with_rank_0_per_experiment[_id]}",
            ]
        _md_file_lines += ["```"]
        _md_file.write_text(
            "\n".join(_md_file_lines)
        )
        print("****************************** END *************************************")
        print("")
        print("")


def main():
    print("*******************************************************************************")
    print("*******************************", sys.argv)
    print("*******************************************************************************")

    experiment = Experiment[sys.argv[1]]  # type: Experiment
    experiment_id = int(sys.argv[2])
    train_or_plot = sys.argv[3]
    if train_or_plot == 'train':
        experiment.train_and_rank(experiment_id)
    elif train_or_plot == 'plot':
        experiment.results()
    else:
        raise Exception(f"Unknown {train_or_plot}")

    print()
    print()
    print()


if __name__ == '__main__':
    main()
