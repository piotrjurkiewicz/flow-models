from setuptools import setup
import flow_models

URL = 'https://github.com/piotrjurkiewicz/flow-models'

with open('README.md') as f:
    README = f.read()
    README = README[0:README.find('## Models library') + 18] + '\n'
    README += f"The repository of flow models, containing histogram CSV files, fitted mixture models and plots, is available at: {URL}"

setup(
    name='flow_models',
    version=flow_models.__version__,
    packages=['flow_models', 'flow_models.lib'],
    url=URL,
    license='MIT',
    author='Piotr Jurkiewicz',
    author_email='piotr.jerzy.jurkiewicz@gmail.com',
    description='A framework for analysis and modeling of IP network flows',
    long_description=README,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'flow-models-merge=flow_models.merge:main',
            'flow-models-sort=flow_models.sort:main',
            'flow-models-hist=flow_models.hist:main',
            'flow-models-hist_np=flow_models.hist_np:main',
            'flow-models-fit=flow_models.fit:main',
            'flow-models-plot=flow_models.plot:main',
            'flow-models-generate=flow_models.generate:main',
            'flow-models-summary=flow_models.summary:main',
            'flow-models-convert=flow_models.convert:main'
        ]
    },
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
