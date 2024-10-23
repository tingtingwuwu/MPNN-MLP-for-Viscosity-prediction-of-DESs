from setuptools import setup, find_packages

# Read the contents of the README file to use as the long description
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='MPNN-MLP-ViscosityPrediction',
    version='0.1.0',
    author='Ting Wu',
    author_email='2112306217@mail2.gdut.edu.cn',
    description='A project for predicting the viscosity of Deep Eutectic Solvents (DESs) using MPNN and MLP models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tingtingwuwu/MPNN-MLP-ViscosityPrediction',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy>=1.26.4',
        'pandas>=2.2.3',
        'scikit-learn>=1.0.2',
        'torch>=2.3.0',
        'torch-geometric>=1.7.0',
        'rdkit-pypi>=2024.03.5',  # RDKit for handling chemical structures
        'shap>=0.31.0',
        'optuna>=2.0.0',
        'matplotlib>=3.9.2',
        'networkx>=3.2.1'
    ],
    extras_require={
        'dev': [
            'pytest>=5.0',
            'flake8>=3.8.0',
            'black>=20.8b1'
        ],
    },
    entry_points={
        'console_scripts': [
            'train_model=scripts.train_model:main',
            'optimize_hyperparameters=scripts.optimize_hyperparameters:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    python_requires='>=3.7',
)

