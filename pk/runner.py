import sys
sys.path.append("..")
import enum
import pathlib
import numpy as np
import typing as t
from keras.models import load_model
import plotly.graph_objects as go
import pickle
import plotly.express as px
import sys
import tqdm
import pandas as pd
import os
import tensorflow as tf
import h5py

ROOT_DIR = pathlib.Path(__file__).parent / "results"
NUM_ATTACKS_PER_EXPERIMENT = 100
NUM_EXPERIMENTS = 100

AES_sbox=[
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
]


def load_ascad_v1_fk(ascad_database_file):
    f  = h5py.File(ascad_database_file, "r")

    X_profiling = f['Profiling_traces/traces'][()]
    Y_profiling = f['Profiling_traces/labels'][()]

    X_attack = f['Attack_traces/traces'][()]
    meta_attack = f['Attack_traces/metadata'][()]

    # Attack traces all have the same key
    real_key = meta_attack['key'][0,:]
    pt_attack = meta_attack['plaintext']

    print("Creating all target labels for the attack traces:")
    targets = np.zeros((X_attack.shape[0], 256), dtype='uint8')
    for i in tqdm.trange(X_attack.shape[0]):
        for k in range(256):
            targets[i, k] = AES_sbox[k^pt_attack[i,2]]

    f.close()

    return X_profiling, Y_profiling, X_attack, targets, real_key[2]


def preprocess_predictions(predictions, all_guess_targets, num_examples, num_guesses) -> np.ndarray:

    # make copy
    predictions = predictions.copy()

    # Add small positive value
    # note that we set any o or negative probability to smallest
    # possible positive number so that np.log does not
    # result to -np.inf
    predictions[predictions <= 1e-45] = 1e-45

    # Sort based on guessed targets
    sorted_predictions = predictions[
        np.asarray(
            [np.arange(num_examples)]
        ).repeat(num_guesses, axis=0).T,
        all_guess_targets
    ]

    # take negative logs
    sorted_neg_log_preds = -np.log(sorted_predictions)

    # return
    return sorted_neg_log_preds


def compute_ranks(predictions, all_guess_targets, correct_key, num_attacks) -> np.ndarray:

    # num_examples and num_guesses
    num_examples = predictions.shape[0]
    num_guesses = 256

    # some buffers
    all_ranks = np.zeros((num_attacks, num_examples), np.uint8)

    # fix seed for deterministic behaviour
    np.random.seed(123456)

    # get sorted_neg_log_preds
    sorted_neg_log_preds = preprocess_predictions(predictions, all_guess_targets, num_examples, num_guesses)

    # loop over
    for attack_id in tqdm.trange(num_attacks):
        # first shuffle for simulating random experiment
        np.random.shuffle(sorted_neg_log_preds)

        # cum sum
        sorted_neg_log_preds_cum_sum = np.cumsum(sorted_neg_log_preds, axis=0)

        # compute rank
        ranks_for_all_guesses = sorted_neg_log_preds_cum_sum.argsort().argsort()

        # set correct rank
        all_ranks[attack_id, :] = ranks_for_all_guesses[:, correct_key]

    # return
    return all_ranks


class Experiment(enum.Enum):

    ascad_v1_fk_0_mlp = enum.auto()
    ascad_v1_fk_50_mlp = enum.auto()
    ascad_v1_fk_100_mlp = enum.auto()
    # ascad_v1_fk_0_cnn = enum.auto()
    # ascad_v1_fk_50_cnn = enum.auto()
    # ascad_v1_fk_100_cnn = enum.auto()

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

        if self.name.startswith("ascad_v1_fk"):
            _database = "../ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/"
            if self.name.find("_0_") != -1:
                _database += "ASCAD.h5"
            elif self.name.find("_50_") != -1:
                _database += "ASCAD_desync50.h5"
            elif self.name.find("_100_") != -1:
                _database += "ASCAD_desync100.h5"
            else:
                raise Exception(f"Unsupported {self}")
        else:
            raise Exception(f"Unsupported {self}")

        return _database

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
                "training_model": (self.store_dir(experiment_id) / "model.hdf5").resolve().as_posix(),
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

    # noinspection PyPep8Naming
    def train(self, experiment_id: int, use_gpu: bool, force: bool = False):
        # so that training happens on cpu ;)
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            assert not tf.test.gpu_device_name()
        # files that will be saved
        _store_dir = self.store_dir(experiment_id)
        _model_file = _store_dir / "model.hdf5"
        _history_file = _store_dir / "history.pickle"
        _train_params_file = _store_dir / "train_params.txt"

        # if model present return
        if _model_file.exists():
            if force:
                _model_file.unlink()
                if _history_file.exists():
                    _history_file.unlink()
            else:
                return

        # make params files
        _train_params_file.touch(exist_ok=False)
        _train_params_file.write_text(str(self.get_train_params(experiment_id=experiment_id)))

        # call
        print("***************************************************************")
        print("********************** TRAINING *******************************")
        print("***************************************************************")
        os.system(
            f"C:/Python38/python ../ASCAD_train_models.py {_train_params_file.as_posix()}")

    def ranks(self, force: bool = False):
        # so that ranking happens on cpu ;)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        assert not tf.test.gpu_device_name()

        for experiment_id in range(NUM_EXPERIMENTS):
            print(f"Computing ranks for experiment {experiment_id} ...")

            # files that will be saved
            _store_dir = self.store_dir(experiment_id)
            _model_file = _store_dir / "model.hdf5"
            _ranks_file = _store_dir / "ranks.npy"

            # if result present return
            if _ranks_file.exists():
                if force:
                    _ranks_file.unlink()
                else:
                    return

            # if model does not exist raise error
            if not _model_file.exists():
                print(f"  > There is no model file for experiment {experiment_id} so skipping ...")
                return

            # get data
            if self.name.startswith("ascad_v1_fk"):
                _data = load_ascad_v1_fk(self.get_database_file_name())
            else:
                raise Exception(f"Experiment {self} is not supported")
            X_profiling, Y_profiling, X_attack, targets, key_attack = _data

            # load model
            _model = load_model(_model_file.as_posix())

            input_layer_shape = _model.get_layer(index=0).input_shape
            if len(input_layer_shape) == 2:
                tracesAttack_shaped = X_attack
            elif len(input_layer_shape) == 3:
                tracesAttack_shaped = X_attack.reshape(
                    (X_attack.shape[0], X_attack.shape[1], 1))
            else:
                raise Exception(f"Unknown shape {len(input_layer_shape)}")

            print('Get predictions:')
            predictions = _model.predict(tracesAttack_shaped, verbose=1)

            print('Evaluating the model:')
            ranks = compute_ranks(
                predictions=predictions,
                all_guess_targets=targets,
                correct_key=key_attack,
                num_attacks=NUM_ATTACKS_PER_EXPERIMENT,
            )

            # Calculate the mean of the rank over the nattack attacks
            avg_rank = np.mean(ranks, axis=0)

            print(np.where(avg_rank <= 0.))

            # save ranks
            np.save(_ranks_file, ranks)

    def results(self):

        # create figures
        avg_rank_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(
                    text=f"{self.name}: Average Rank")
            )
        )
        rank_variance_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(
                    text=f"{self.name}: Rank Variance")
            )
        )
        loss_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{self.name}: Train Loss")
            )
        )
        accuracy_fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"{self.name}: Accuracy")
            )
        )

        # plot
        non_converged_experiments = []
        min_traces_for_avg_rank_0 = []
        for experiment_id in range(NUM_EXPERIMENTS):

            # get store dir
            store_dir = self.store_dir(experiment_id)
            history_file = store_dir / f"history.pickle"
            ranks_file = store_dir / f"ranks.npy"
            # skip is results not present
            if not history_file.exists() or not ranks_file.exists():
                continue

            # history
            with open((store_dir / f"history.pickle").as_posix(), 'rb') as file_pi:
                history = pickle.load(file_pi)
            loss = history['loss']
            accuracy = history['accuracy']

            # ranks
            ranks = np.load(store_dir / f"ranks.npy")
            avg_rank = np.mean(ranks, axis=0)
            rank_variance = np.var(ranks, axis=0)
            traces_with_rank_0 = np.where(avg_rank <= 0)[0]
            if len(traces_with_rank_0) == 0:
                non_converged_experiments.append(experiment_id)
                min_traces_for_avg_rank_0.append(np.inf)
            else:
                min_traces_for_avg_rank_0.append(traces_with_rank_0[0])

            # noinspection DuplicatedCode
            rank_plot_until = self.rank_plot_until
            avg_rank_fig.add_trace(
                go.Scatter(
                    x=np.arange(rank_plot_until),
                    y=avg_rank[:rank_plot_until],
                    mode='lines',
                    name=f"exp_{experiment_id:03d}"
                )
            )
            rank_variance_fig.add_trace(
                go.Scatter(
                    x=np.arange(rank_plot_until),
                    y=rank_variance[:rank_plot_until],
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
        df['min_traces_for_avg_rank_0'] = [
            _ for _ in min_traces_for_avg_rank_0 if _ != np.inf
        ]
        hist_fig = px.histogram(
            df, x="min_traces_for_avg_rank_0",
            title=f"{self.name}: Histogram",
        )
        hist_fig.write_image((self.plot_dir / f"hist.svg").as_posix())
        print("plotting average rank")
        avg_rank_fig.write_image((self.plot_dir / f"average_rank.svg").as_posix())
        print("plotting rank variance")
        rank_variance_fig.write_image((self.plot_dir / f"rank_variance.svg").as_posix())
        print("plotting train loss")
        loss_fig.write_image((self.plot_dir / f"loss.svg").as_posix())
        print("plotting accuracy")
        accuracy_fig.write_image((self.plot_dir / f"accuracy.svg").as_posix())

        # write details to md file
        _md_file = self.plot_dir.parent / f"{self.name}.md"
        _md_file_lines = [
            f"# Results for {self.name}", "", f"## Convergence", ""
        ]
        if bool(non_converged_experiments):
            _percent_failed = (len(non_converged_experiments) / NUM_EXPERIMENTS) * 100.
            _md_file_lines += [
                f"**Note that `{_percent_failed} %` experiments did not converge.**",
                f"Failed experiment id's are as below:",
                f"  > {non_converged_experiments}", ""
            ]
        else:
            _md_file_lines += [
                f"Note that all {NUM_EXPERIMENTS} experiments converged", ""
            ]
        _md_file_lines += [
            f"## Histogram",
            "",
            f"![Histogram - {self.name}]({self.name}/hist.svg)",
            "",
            f"## Average Rank",
            "",
            f"![Average Rank - {self.name}]({self.name}/average_rank.svg)",
            "",
            f"## Rank Variance",
            "",
            f"![Rank Variance - {self.name}]({self.name}/rank_variance.svg)",
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
        _sorted_ids = np.argsort(min_traces_for_avg_rank_0)
        _md_file_lines += [f"## Sorted Experiment IDs with min traces", "", "```"]
        for _id in _sorted_ids:
            _md_file_lines += [
                f"experiment {_id:03d} :: min traces {min_traces_for_avg_rank_0[_id]}",
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
    _mode = sys.argv[2]
    if _mode == 'train':
        _use_gpu = False
        _experiment_id = int(sys.argv[3])
        try:
            if sys.argv[4] == "use_gpu":
                _use_gpu = True
            else:
                raise Exception(f"Unknown sys argv {sys.argv[4]}")
        except IndexError:
            ...
        experiment.train(_experiment_id, use_gpu=_use_gpu)
    elif _mode == 'ranks':
        experiment.ranks()
    else:
        raise Exception(f"Unknown {_mode}")

    print()
    print()
    print()


if __name__ == '__main__':
    main()
