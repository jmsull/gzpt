from setuptools import setup
from io import open

# read the contents of the README file
with open('README.md', "r") as f:
    long_description = f.read()

def find_version(path):
    with open(path, 'r') as fp:
        file = fp.read()
    import re
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                            file, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Version not found")

setup(name='gzpt',
      version = find_version("gzpt/__init__.py"),
      description='Hybrid Analytic Model for Galaxy Power Spectrum',
      url='https://github.com/jmsull/gzpt',
      author='James M Sullivan',
      author_email='jmsullivan@berkeley.edu',
      license='MIT',
      packages=['gzpt'],
      install_requires=['wheel', 'numpy', 'scipy','pyfftw'],
      tests_require=['numpy','scipy'],
      extras_require={
        'testing':  ["numpy"],
        },
      python_requires='>=3',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Physics'
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      keywords='cosmology')