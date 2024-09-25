from setuptools import setup
import flow_models

URL = 'https://github.com/piotrjurkiewicz/flow-models'

with open('README.md') as f:
    README = f.read()
    README = README[0:README.find('## Models library') + 18] + '\n'
    README += f"The repository of flow models, containing histogram CSV files, fitted mixture models, plots, and full flow records in case of smaller models is available at: {URL}#models-library"

setup(
    name='flow_models',
    version=flow_models.__version__,
    packages=['flow_models', 'flow_models.lib', 'flow_models.first_mirror', 'flow_models.elephants'],
    url=URL,
    license='MIT',
    author='Piotr Jurkiewicz',
    author_email='piotr.jerzy.jurkiewicz@gmail.com',
    description='A framework for analysis and modeling of IP network flows',
    long_description=README,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Topic :: Internet",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: System :: Networking",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Telecommunications Industry"
    ]
)
