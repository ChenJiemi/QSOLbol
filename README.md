Project: QSOLbol

Author:JieChen

Email: jiechen@stu.pku.edu.cn

# QSOLbol

QSOLbol is a Python tool to estimate the bolometric luminosity of quasars from multi-wavelength data (rest-frame SEDs) with SOM.


# install 
Currently the data file content is too large (QSOLbol/config/data), temporarily stored in https://disk.pku.edu.cn/link/AAEA9B507C30CA4FB395820FC498A35BCA (extraction code: qF51). Please add lbol_data.npz into QSOLbol/config/data.

cd ./QSOLbol
pip install .

# import
from QSOLbol.core import QSOLbol

calculator= QSOLbol()

results= calculator.calculate(wave, mags, mags_err, z,scale=True)

logLbol=results[0]

logLbol_err=results[1]


