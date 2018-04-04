# Google Landmark Recognition Challenge

## Setup

### Virtualenv

```
conda create -n landmark-recognition-challenge python=3.6
source activate landmark-recognition-challenge

conda install pytorch torchvision cuda90 -c pytorch  # Note: change cuda depending on your version
conda install jupyter scikit-learn matplotlib pandas
```

### Downloading the data

```
kaggle competitions download -c landmark-recognition-challenge --path=data

for file in $(ls data/*zip); do unzip $file -d data/; done

python img_downloader.py data/train.csv data/train
```
