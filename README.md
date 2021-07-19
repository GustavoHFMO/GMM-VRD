# GMM-VRD: A Gaussian Mixture Model for Dealing With Virtual and Real Concept Drifts [![DOI](https://zenodo.org/badge/306144251.svg)](https://zenodo.org/badge/latestdoi/306144251)

## Usage
```
# Cloning the repository
git clone https://github.com/GustavoHFMO/GMM-VRD.git

# Acessing the repository
cd GMM-VRD

# Installing the dependencies
pip install -r requirements.txt
```

# GMM with training in batch

The module [GMM_batch.py](https://github.com/GustavoHFMO/GMM-VRD/blob/master/GMM_batch.py) shows how to train a GMM for classification using a batch of observations, and also plots the generated model.

```
# Running the code
python GMM_batch.py
```

## Result
![](https://github.com/GustavoHFMO/GMM-VRD/blob/master/images/gmm_training_set.png)
![](https://github.com/GustavoHFMO/GMM-VRD/blob/master/images/gmm_test_set.png)

# GMM-VRD to handle virtual and real drifts

The module [GMM_online.py](https://github.com/GustavoHFMO/GMM-VRD/blob/master/GMM_online.py) executes the algorithms described below in real and synthetic datasets.

```
# Running the code
python GMM_online.py
```

## [GMM-VRD:](https://github.com/GustavoHFMO/GMM-VRD/blob/master/competitive_algorithms/gmm_vrd.py)
> Oliveira, Gustavo HFM, Leandro L. Minku, and Adriano LI Oliveira. "GMM-VRD: A Gaussian Mixture Model for Dealing With Virtual and Real Concept Drifts." 2019 International Joint Conference on Neural Networks (IJCNN). IEEE, 2019.

## [Dynse:](https://github.com/GustavoHFMO/GMM-VRD/blob/master/competitive_algorithms/dynse.py)
> P. R. Almeida, L. S. Oliveira, A. S. Britto Jr, and R. Sabourin, “Adapting dynamic classifier selection for concept drift,” Expert Systems with Applications, vol. 104, pp. 67–85, 2018.

## [IGMM-CD:](https://github.com/GustavoHFMO/GMM-VRD/blob/master/competitive_algorithms/igmmcd.py)
> L. S. Oliveira and G. E. Batista, “Igmm-cd: a gaussian mixture classification algorithm for data streams with concept drifts,” in BRACIS, 2015 Brazilian Conference on. IEEE, 2015, pp. 55–61

## Result
![](https://github.com/GustavoHFMO/GMM-VRD/blob/master/images/gmm_vrd_execution.png)

## License
This project is under a GNU General Public License (GPL) Version 3. See [LICENSE](https://www.gnu.org/licenses/gpl-3.0-standalone.html) for more information.
