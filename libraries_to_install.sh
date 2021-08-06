!/bin/bash

#update and upgrade
apt install update
apt install upgrade -y

#install pip
apt install python3-pip

#install packages
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install sklearn
pip install joblib
pip install install pyexcel-xlsx
pip install XlsxWriter
pip install xlrd
pip install openpyxl

#run automated command
bash create_model.sh