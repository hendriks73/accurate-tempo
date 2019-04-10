from setuptools import setup, find_packages

import glob

# define scripts to be installed by the PyPI package
scripts = glob.glob('bin/*')

setup(name='accuratetempo',
      version='0.1',
      packages=find_packages(),
      description='High-Accuracy Musical Tempo Estimation using Convolutional Neural Networks and Autocorrelation',
      author='anon',
      author_email='anyn',
      license='Attribution 3.0 Unported (CC BY 3.0)',
      scripts=scripts,
      install_requires=[
          'h5py',
          'scikit-learn',
          'librosa',
          'tensorflow-gpu==1.10.1;platform_system!="Darwin"',
          'tensorflow==1.10.1;platform_system=="Darwin"',
          'pytest',
          'matplotlib',
          'jams',
      ],
      include_package_data=True,
      zip_safe=False)
