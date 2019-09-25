from setuptools import setup, Extension, find_packages

setup(
    name='svd_lsq',
    version='1.0',
    packages=find_packages(),
    package_dir={'svd_lsq': './src/svd_lsq'},
    ext_modules=([Extension('_svd_lsq',
                            ['./src/cpp/svd_lsq.cpp'],
                            include_dirs=["/usr/include/eigen3/"])])
)
