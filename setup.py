from setuptools import setup, find_packages

setup(
    name='tkgdti',
    version='0.1.0',
    author='Nathaniel Evans',
    author_email='evansna@ohsu.edu',
    description='Targetome Knowledge Graph for Drug-Target Interaction Prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/biodev/TKG-DTI',
    license='GPL-3.0',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='drug-target-interaction, knowledge-graph, graph-neural-network, bioinformatics',
)