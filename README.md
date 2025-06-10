Project: QSOLbol

Author:JieChen

Email: jiechen@stu.pku.edu.cn

# QSOLbol

QSOLbol is a Python tool to estimate the bolometric luminosity of quasars from multi-wavelength data (rest-frame SEDs) with SOM.


# install 
Currently the data file content is too large (QSOLbol/config/data), temporarily stored in https://disk.pku.edu.cn/link/AAEA9B507C30CA4FB395820FC498A35BCA (extraction code: qF51). Please add lbol_data.npz into QSOLbol/config/data.

cd ./QSOLbol
pip install .

# Parameters
| **Parameters**   | **Type**         | **Description**                                                                 |
|------------------|------------------|---------------------------------------------------------------------------------|
| `wave`           | (n_sample,n_SED)      | Observed-frame SED wavelength in Angstrom.                                      |
| `mags`           | (n_sample,n_SED)     | Observed-frame SED in AB magnitude.                                            |
| `mags_err`       | (n_sample,n_SED)      | Observed-frame SED error in AB magnitude.                                      |
| `z`              | (n_sample,1)    | Redshift.                                                                      |
| `f_isotropy`     | bool             | Whether to use 0.75 correction for viewing angle. The default is `False`.     |
| `wave_range`     | tuple            | The integrated range of bolometric luminosity in Hz. The default is from 4 µm to 2 keV. |
| scale            | bool             | Whether to scale the SED when the source is at faint and bright luminosity ends. |

# import
from QSOLbol.core import QSOLbol

calculator= QSOLbol()

results= calculator.calculate(wave, mags, mags_err, z,scale=True)

logLbol=results[0]

logLbol_err=results[1]


