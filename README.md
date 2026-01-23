# Continuous-Time Transfer Entropy Estimation for Point Processes via Recurrent Mixture Density Networks: Application to Spike Trains

Modified from Pytorch implementation of the paper ["Intensity-Free Learning of Temporal Point Processes"](https://openreview.net/forum?id=HygOjhEYDH), Oleksandr Shchur, Marin Biloš and Stephan Günnemann, ICLR 2020.

## Usage
In order to run the code, you need to install the `dpp` library.
```bash
cd code
python setup.py install
```

The main estimation code is `\code\entropy_tpp.py`. To compare with CoTETE estimator use `\code\CoTETE_example_test.py`


## Requirements
```
python=3.12.9
numpy=2.2.5
pytorch=2.5.1
pytorch-cuda=12.4 (optional)
scikit-learn=1.6.1
scipy=1.15.3
optuna=4.4.0
pandas=2.2.3
juliacall=0.9.28 (optional)
```


## Cite
```
@article{
    shchur2020intensity,
    title={Intensity-Free Learning of Temporal Point Processes},
    author={Oleksandr Shchur and Marin Bilo\v{s} and Stephan G\"{u}nnemann},
    journal={International Conference on Learning Representations (ICLR)},
    year={2020},
}
```
