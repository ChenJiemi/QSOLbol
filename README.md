Project: QSOLbol
Author:JieChen
Email: jiechen@stu.pku.edu.cn

# QSOLbol

QSOLbol is a Python tool to estimate the bolometric luminosity of quasars from multi-wavelength data (rest-frame SEDs) with SOM.

# install 
cd ./QSOLbol
pip install .

# import
from QSOLbol.core import QSOLbol

calculator= QSOLbol()

results= calculator.calculate(wave, mags, mags_err, z,scale=True)

logLbol=results[0]

logLbol_err=results[1]


