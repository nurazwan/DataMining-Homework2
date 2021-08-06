!/bin/bash

#update and upgrade
apt update
apt upgrade -y

#install pip
apt install python3-pip -y

#install packages
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install seaborn
pip3 install sklearn
pip3 install joblib
pip3 install install pyexcel-xlsx
pip3 install XlsxWriter
pip3 install xlrd
pip3 install openpyxl

#run automated command
bash create_model.sh
