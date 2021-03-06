#!/bin/bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzvf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install
python3 -m pip install Ta-Lib
git clone --recursive http://github.com/dmlc/xgboost
cd xgboost
make
cd python-package
python3 setup.py install
apt remove --purge cmake
hash -r
snap install cmake --classic
python3 -m pip install xgboost                   