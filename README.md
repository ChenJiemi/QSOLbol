# QSOLbol

QSOLbol is a Python tool to estimate the bolometric luminosity of quasars from multi-wavelength data (rest-frame SEDs) with SOM.
The SOM package is from ([https://github.com/sevamoo/SOMPY]). Please install the SOMPY before you star QSOLbol.

# install 
```
cd ./QSOLbol
pip install .
```

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
```
from QSOLbol.core import QSOLbol

calculator= QSOLbol()

results= calculator.calculate(wave, mags, mags_err, z,scale=True)

logLbol=results[0]

logLbol_err=results[1]
```

# Citation
Please cite the paper if you used the QSOLbol, as follows ([https://iopscience.iop.org/article/10.3847/1538-4357/ade307
]):

```
@ARTICLE{2025ApJ...988..204C,
       author = {{Chen}, Jie and {Jiang}, Linhua and {Sun}, Shengxiu and {Zhang}, Zijian and {Sun}, Mouyuan},
        title = "{Estimating Bolometric Luminosities of Type 1 Quasars with Self-organizing Maps}",
      journal = {\apj},
     keywords = {Quasars, Surveys, Neural networks, Spectral energy distribution, 1319, 1671, 1933, 2129, Cosmology and Nongalactic Astrophysics, Astrophysics of Galaxies, High Energy Astrophysical Phenomena},
         year = 2025,
        month = aug,
       volume = {988},
       number = {2},
          eid = {204},
        pages = {204},
          doi = {10.3847/1538-4357/ade307},
archivePrefix = {arXiv},
       eprint = {2506.04329},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...988..204C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
```

Author: Jie Chen (jiechen(at)stu.pku.edu.cn)
