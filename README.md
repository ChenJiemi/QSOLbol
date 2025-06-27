# QSOLbol

QSOLbol is a Python tool to estimate the bolometric luminosity of quasars from multi-wavelength data (rest-frame SEDs) with SOM.


# install 
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
| `scale`          | bool             | Whether to scale the SED when the source is at faint and bright luminosity ends. The default is `False`.|



# import
from QSOLbol.core import QSOLbol

calculator= QSOLbol()

results= calculator.calculate(wave, mags, mags_err, z,scale=True)

logLbol=results[0]

logLbol_err=results[1]

# Citation

@ARTICLE{2025arXiv250604329C,
       author = {{Chen}, Jie and {Jiang}, Linhua and {Sun}, Shengxiu and {Zhang}, Zijian and {Sun}, Mouyuan},
        title = "{Estimating Bolometric Luminosities of Type 1 Quasars with Self-Organizing Maps}",
      journal = {arXiv e-prints},
     keywords = {Cosmology and Nongalactic Astrophysics, Astrophysics of Galaxies, High Energy Astrophysical Phenomena},
         year = 2025,
        month = jun,
          eid = {arXiv:2506.04329},
        pages = {arXiv:2506.04329},
          doi = {10.48550/arXiv.2506.04329},
archivePrefix = {arXiv},
       eprint = {2506.04329},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250604329C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}



Author: JieChen

Email: jiechen@stu.pku.edu.cn
