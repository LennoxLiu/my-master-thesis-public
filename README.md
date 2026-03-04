# Continuous-Time Transfer Entropy Estimation for Point Processes via Recurrent Mixture Density Networks: Application to Spike Trains

This project developed a deep learning framework to estimate the transfer entropy rate in continuous time for point processes like spike trains.

Modified from Pytorch implementation of the paper ["Intensity-Free Learning of Temporal Point Processes"](https://openreview.net/forum?id=HygOjhEYDH), Oleksandr Shchur, Marin Biloš and Stephan Günnemann, ICLR 2020.

## Usage
Check out `demo/demo.ipynb` for the demonstration of whole estimation process.


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
