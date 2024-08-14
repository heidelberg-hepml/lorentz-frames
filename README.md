# lorentz-frames

We adapt the tensorframes of https://arxiv.org/abs/2405.15389 to the Lorentz group.

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

The datasets can be downloaded from the Heidelberg ITP website ([toptagging](https://www.thphys.uni-heidelberg.de/~plehn/data/toptagging_full.npz)). Finally, keys in the`data` section of the config files have to be adapted to specify where the datasets are located on your machine (`data_path` or `data_dir` depending on the experiment).

## 2. Running experiments

You can run experiments with the following commands:
```bash
python run.py -cn toptagging model=gcnconv exp_name=toptagging run_name=hello_world_toptagging
```

We use hydra for configuration management, allowing to quickly override parameters in e.g. config/toptagging.yaml. Further, we use mlflow for tracking. You can start a mlflow server based on the saved results in runs/tracking/mlflow.db on port 4242 of your machine with the following command

```bash
mlflow ui --port 4242 --backend-store-uri sqlite:///runs/tracking/mlflow.db
```

An existing run can be reloaded to perform additional tests with the trained model. For a previous run with exp_name=toptagging and run_name=hello_world_toptagging, one can run for example. 
```bash
python run.py -cn config -cp runs/toptagging/hello_world_toptagging train=false warm_start_idx=0
```
The warm_start_idx specifies which model in the models folder should be loaded and defaults to 0. 

The default configuration files in the `config` folder define small models to allow quick test runs.