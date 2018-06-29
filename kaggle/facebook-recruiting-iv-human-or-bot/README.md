# Facebook Recruiting IV Human or Bot

## Setup

```
# Setup virtualenv
conda create -n human-or-bot python=3.6 -y
source activate human-or-bot
conda install pandas matplotlib -y
conda install -c anaconda jupyter scikit-learn -y

# Install LightGBM
git clone --recursive https://github.com/Microsoft/LightGBM.git
cd LightGBM/python-package
# export CXX=g++-8 CC=gcc-8  # for macOS users only (replace 8 with version of gcc installed on your machine)
python setup.py install
```
