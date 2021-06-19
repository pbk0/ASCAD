import pathlib
import sys
import runner
import gc
import os


def wipe_incomplete():
    for i in range(100):
        print(f"Experiment {i:03d}")
        gc.collect()
        for experiment in runner.Experiment:
            _log_path = pathlib.Path(f'results/{experiment.name}/{i}/log.txt')
            if _log_path.exists():
                if not experiment.exists(i):
                    print(f"  > {experiment.name}:{i:03d} ... wiping")
                    for _ in _log_path.parent.iterdir():
                        _.unlink()


def train_all():
    for i in range(100):
        print(f"Experiment {i:03d}")
        gc.collect()
        for experiment in runner.Experiment:
            _log_path = pathlib.Path(f'results/{experiment.name}/{i}/log.txt')
            _exists = experiment.exists(i)
            if _log_path.exists():
                if _exists:
                    print(f"  > {experiment.name}:{i:03d} ... results already available")
                else:
                    print(f"  > {experiment.name}:{i:03d} ... someone else is working")
            else:
                print(f"  > {experiment.name}:{i:03d}")
                if _exists:
                    raise Exception(f"Where is the log file ??")
                else:
                    os.system(
                        f"C:/Python38/python runner.py {experiment.name} {i} train > "
                        f"{_log_path.as_posix()}")


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


def main():
    _mode = sys.argv[1]
    if _mode == 'train':
        train_all()
    elif _mode == 'wipe':
        wipe_incomplete()
    elif _mode == 'show':
        show_results()
    else:
        raise Exception(f"Unknown {_mode}")


if __name__ == '__main__':
    main()



