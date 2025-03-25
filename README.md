# lorentz-frames

We adapt the tensorframes of https://arxiv.org/abs/2405.15389 to the Lorentz group and benchmark it on amplitude regression and jet tagging.

## 1. Getting started

Clone the repository.

```bash
git clone https://github.com/heidelberg-hepml/lorentz-frames
```

Create a virtual environment and install requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Small test datasets are contained in the `data/` directory. The full datasets can be downloaded from the Heidelberg ITP website ([amplitudes](https://www.thphys.uni-heidelberg.de/~plehn/data/amplitudes.hdf5), [toptagging](https://www.thphys.uni-heidelberg.de/~plehn/data/toptagging_full.npz)) using the script `data/collect_data.py`. Finally, keys in the`data` section of the config files have to be adapted to specify where the datasets are located on your machine (`data_path` or `data_dir` depending on the experiment).

## 2. Running tests

Most parts of the code are covered with unit tests. Before running any experiments, you can check that the code is healthy by running these tests

```bash
pytest tests
```

If they all pass you are good. If not, it either means that there is a problem with your environment, or that someone pushed changes that made the tests crash. In any case, you should get these tests to pass before moving on to running the main code. Note that `pytest` gives you a lot of freedom in controlling these tests, e.g. you can run specific tests and use the `-s` modifier to print all outputs

```bash
pytest tests/utils/test_orthogonalize.py::test_lorentz_cross -s
```

## 3. Running experiments

You can run a quick test toptagging experiment with the following command

```bash
python run.py
```

We use hydra for configuration management, allowing to quickly override parameters in e.g. config_quick/toptagging.yaml. Configuration files for small test runs are in `config_quick`, if you want to run the big runs you should use `-cn config`. The `model`, `training`, `lframesnet` and `equivectors` option can be selected as follows, and individual keys can be modified with the `.` operator

```bash
python run.py -cp config -cn toptagging model=graphnet training=graphnet model/lframesnet=orthogonal model/lframesnet/equivectors=equigraph training.iterations=42
```

Further, we use mlflow for tracking. You can start a mlflow server based on the saved results in runs/tracking/mlflow.db on port 4242 of your machine with the following command

```bash
mlflow ui --port 4242 --backend-store-uri sqlite:///runs/tracking/mlflow.db
```

An existing run can be reloaded to perform additional tests with the trained model. For a previous run with exp_name=toptagging and run_name=hello_world_toptagging, one can run for example

```bash
python run.py -cn config -cp runs/toptagging/hello_world_toptagging train=false warm_start_idx=0
```

The warm_start_idx specifies which model in the models folder should be loaded and defaults to 0. 