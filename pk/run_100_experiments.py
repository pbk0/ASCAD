import pathlib
import sys
import runner
import gc
import os
import zipfile


def wipe_file(file_name):
    for i in range(runner.NUM_EXPERIMENTS):
        print(f"Experiment {i:03d} ... wiping {file_name}")
        for experiment in runner.Experiment:
            _file_path = pathlib.Path(f'results/{experiment.name}/{i}/{file_name}')
            if _file_path.exists():
                _file_path.unlink()


def train_all():
    for i in range(runner.NUM_EXPERIMENTS):
        print(f"Experiment {i:03d}")
        gc.collect()
        for experiment in runner.Experiment:

            # check if results exist
            _model_file = experiment.store_dir(i) / "model.hdf5"
            if _model_file.exists():
                continue

            # make semaphore
            _semaphore_file = pathlib.Path(f'results/{experiment.name}/{i}/__computing__')
            if _semaphore_file.exists():
                print(f"  > {experiment.name}:{i:03d} ... someone else is working")
            else:
                _semaphore_file.touch()
                os.system(
                    f"C:/Python38/python runner.py {experiment.name} {i} train")
                _semaphore_file.unlink()


def ranks_all():
    for i in range(runner.NUM_EXPERIMENTS):
        print(f"Experiment {i:03d}")
        gc.collect()
        for experiment in runner.Experiment:

            # check if results exist
            _ranks_file = experiment.store_dir(i) / "ranks.npy"
            if _ranks_file.exists():
                continue

            # make semaphore
            _semaphore_file = pathlib.Path(f'results/{experiment.name}/{i}/__computing__')
            if _semaphore_file.exists():
                print(f"  > {experiment.name}:{i:03d} ... someone else is working")
            else:
                _semaphore_file.touch()
                os.system(
                    f"C:/Python38/python runner.py {experiment.name} {i} ranks")
                _semaphore_file.unlink()


def show_results():
    _plots_md = runner.ROOT_DIR / "plots.md"
    _plots_md_lines = [
        f"# Plots for 100 experiments", ""
    ]
    for _e in runner.Experiment:
        _plots_md_lines += [
            f"+ [{_e.name}](plots/{_e.name}.md)"
        ]
        _e.results()
    _plots_md.write_text(
        "\n".join(_plots_md_lines)
    )


def zip_results():
    _zip_file_path = pathlib.Path("results_ascad.zip")
    if _zip_file_path.exists():
        _zip_file_path.unlink()
    _zip_file = zipfile.ZipFile("results_ascad.zip", 'w')
    for i in range(runner.NUM_EXPERIMENTS):
        for experiment in runner.Experiment:
            store_dir = experiment.store_dir(i)
            history_file = store_dir / f"history.pickle"
            if history_file.exists():
                _zip_file.write(history_file, arcname=f"results/{experiment.name}/{i}/history.pickle")
            ranks_file = store_dir / f"ranks.npy"
            if ranks_file.exists():
                _zip_file.write(ranks_file, arcname=f"results/{experiment.name}/{i}/ranks.npy")
    _zip_file.close()


def main():
    _mode = sys.argv[1]
    if _mode == 'train':
        train_all()
    elif _mode == 'ranks':
        ranks_all()
    elif _mode == 'wipe':
        wipe_file(sys.argv[2])
    elif _mode == 'show':
        show_results()
    elif _mode == 'zip':
        zip_results()
    else:
        raise Exception(f"Unknown {_mode}")


if __name__ == '__main__':
    main()



