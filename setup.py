"""
SpatialCluster package (c) Ryan Martin 2018
"""
from setuptools import setup


def get_version():
    with open('spatialcluster/__init__.py') as init:
        for line in init.readlines():
            if line.startswith('__version__'):
                return line.split('=')[1].strip().replace("'", "").replace('"', '')
    raise ValueError('Could not infer version!')


if __name__ == '__main__':

    setup(name='spatialcluster',
          version=get_version(),
          description='Spatial Clustering and related Utilities',
          url='https://bitbucket.org/rdmar/spatialcluster',
          maintainer='Ryan Martin',
          maintainer_email='rdm1@ualberta.ca',
          author=['Ryan Martin'],
          license='GPLv3 / CCG',
          python_requires='>3.6',
          install_requires=['numpy>=1.14', 'scipy', 'scikit-learn', 'numba', 'tqdm',
                            'umap-learn'],
          packages=['spatialcluster', 'spatialcluster.ensemble',
                    'spatialcluster.cmd', 'spatialcluster.examples'],
          entry_points={
              # `python setup.py develop` to test with a dev install
              'console_scripts': [
                  'acens=spatialcluster.cmd.acens:main',
                  'acclus=spatialcluster.cmd.acclus:main',
                  'dssens=spatialcluster.cmd.dssens:main',
                  'spatialclusterex=spatialcluster.examples.startnotebooks:main'
              ]
          },
          package_data={'': ['*.ipynb', '*.dat']},
          zip_safe=False)
