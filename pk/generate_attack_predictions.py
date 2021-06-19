"""
This needs to be update for all trained models provided and upgrade to tf 2.x
"""

import pathlib
import sys
import tensorflow as tf
import keras
import numpy as np
import typing as t
import h5py
import yaml
from keras.models import load_model

DATASETS_DIR = pathlib.Path(".datasets")
MODELS_DIR = pathlib.Path(".trained_models")
INPUTS_DIR = pathlib.Path(".inputs")
INPUTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    'fk': {
        0: DATASETS_DIR / "fk" / "ASCAD.h5",
        50: DATASETS_DIR / "fk" / "ASCAD_desync50.h5",
        100: DATASETS_DIR / "fk" / "ASCAD_desync100.h5",
    },
    'vk': {
        0: DATASETS_DIR / "vk" / "ascad-variable.h5",
        50: DATASETS_DIR / "vk" / "ascad-variable-desync50.h5",
        100: DATASETS_DIR / "vk" / "ascad-variable-desync100.h5",
    },
}

MODELS = {
    'fk': {
        'cnn_desync0':
            MODELS_DIR / "fk" /
            "cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5",
        'cnn_desync50':
            MODELS_DIR / "fk" /
            "cnn_best_ascad_desync50_epochs75_classes256_batchsize200.h5",
        'cnn_desync100':
            MODELS_DIR / "fk" /
            "cnn_best_ascad_desync100_epochs75_classes256_batchsize200.h5",
        'mlp_desync0':
            MODELS_DIR / "fk" /
            "mlp_best_ascad_desync0_node200_layernb6_epochs200_"
            "classes256_batchsize100.h5",
        'mlp_desync50':
            MODELS_DIR / "fk" /
            "mlp_best_ascad_desync50_node200_layernb6_epochs200_"
            "classes256_batchsize100.h5",
        'mlp_desync100':
            MODELS_DIR / "fk" /
            "mlp_best_ascad_desync100_node200_layernb6_epochs200_"
            "classes256_batchsize100.h5",
    },
    'vk': {
        'cnn2_desync0':
            MODELS_DIR / "vk" /
            "cnn2-ascad-desync0.h5",
        'cnn2_desync50':
            MODELS_DIR / "vk" /
            "cnn2-ascad-desync50.h5",
    },
}


def check_compatibility():
    import sys
    if sys.version_info[0] != 3 or sys.version_info[1] != 7:
        raise Exception(
            f"We expect python version 3.7.x"
        )
    if tf.__version__ != '1.13.1':
        raise Exception(
            f"We except tensorflow version 1.5.x, found {tf.__version__}"
        )
    if keras.__version__ != '2.2.4':
        raise Exception(
            f"We expect keras version to be 2.2.4, found {keras.__version__}"
        )


def check_if_all_files_present():
    for f in list(DATASETS['fk'].values()) + list(DATASETS['vk'].values()) + \
             list(MODELS['fk'].values()) + list(MODELS['vk'].values()):
        if not f.exists():
            raise Exception(
                f"We expect file {f} to be present. Make sure that you refer "
                f"to https://github.com/ANSSI-FR/ASCAD to get all necessary "
                f"files."
            )


def get_attack_crypto(fk_or_vk: str, desync: int) -> t.Tuple[
    np.ndarray, t.Dict[str, int]
]:

    data_file = DATASETS[fk_or_vk][desync]

    h5_file = h5py.File(data_file.as_posix(), "r")

    attack_ptx = h5_file['Attack_traces/metadata']['plaintext']

    _key_byte_index = 2
    _correct_key = \
        int(h5_file['Attack_traces/metadata']['key'][0][_key_byte_index])

    return attack_ptx, dict(
        correct_key=_correct_key, key_byte_index=_key_byte_index
    )


def get_attack_trace(fk_or_vk: str, desync: int) -> np.ndarray:

    data_file = DATASETS[fk_or_vk][desync]

    h5_file = h5py.File(data_file.as_posix(), "r")

    attack_trace = np.array(h5_file['Attack_traces/traces'], dtype=np.int8)

    return attack_trace


def main():
    check_compatibility()
    check_if_all_files_present()

    # save attack plaintexts, key_info, predictions to INPUTS dir
    for fk_or_vk in MODELS.keys():
        for model in MODELS[fk_or_vk].keys():
            # create folder for model
            _folder = INPUTS_DIR / f"ASCAD_{fk_or_vk}_{model}"
            _folder.parent.mkdir(exist_ok=True)

            # log
            print(f"Fetching plaintext, key_info and predictions for "
                  f"{_folder.name}")

            # file_paths
            _ptx_file_path = _folder / 'plaintext.npy'
            _key_info_file_path = _folder / 'key.info'
            _predictions_file_path = _folder / 'predictions.npy'

            # if predictions exist then return
            if _predictions_file_path.exists():
                continue

            # get desync level
            _desync = int(model.split("desync")[1])

            # save plaintext and key_info
            _plaintext, _key_info = get_attack_crypto(fk_or_vk, _desync)
            np.save(_ptx_file_path.as_posix(), _plaintext)
            _key_info_file_path.write_text(
                yaml.dump(_key_info, Dumper=yaml.SafeDumper)
            )

            # load model
            _model = load_model(MODELS[fk_or_vk][model].as_posix())

            # input layer shape
            _input_layer_shape = _model.get_layer(index=0).input_shape

            # get attack traces
            _attack_traces = get_attack_trace(fk_or_vk, _desync)

            # adapt the data shape according our model input
            if len(_input_layer_shape) == 2:
                # this is a MLP
                _attack_traces = _attack_traces
            elif len(_input_layer_shape) == 3:
                # This is a CNN: reshape the data
                _attack_traces = _attack_traces.reshape(
                    (_attack_traces.shape[0], _attack_traces.shape[1], 1))
            else:
                raise Exception(
                    f"Unsupported input layer shape {_input_layer_shape}")

            # compute and save predictions
            _predictions = _model.predict(_attack_traces)
            np.save(_predictions_file_path.as_posix(), _predictions)


if __name__ == '__main__':
    main()