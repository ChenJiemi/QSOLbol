from setuptools import setup, find_packages

setup(
    name="QSOLbol",
    version="0.1.0",
    description='Quasar bolometric luminosity estimator',
    author='JieChen',
    author_email='jiechen@stu.pku.edu.cn',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "astropy>=5.0",
        "scikit-learn>=1.0",
        "pyyaml>=6.0", 
    ],
    package_data={
        "QSOLbol": ["config/*.yaml", "data/*.pkl", "data/lbol_err_sim.npz","data/lbol_data.npz"],
    },
    include_package_data=True,  
)

