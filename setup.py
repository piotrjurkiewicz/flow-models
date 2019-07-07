from setuptools import setup

setup(
        name='flow_models',
        version='1.0',
        packages=['flow_models', 'flow_models.lib'],
        url='https://github.com/piotrjurkiewicz/flow-models',
        license='MIT',
        author='Piotr Jurkiewicz',
        author_email='piotr.jerzy.jurkiewicz@gmail.com',
        description='',
        entry_points={
            'console_scripts': ['flow-models-convert=flow_models.convert:main',
                                'flow-models-fit=flow_models.fit:main',
                                'flow-models-hist=flow_models.hist:main',
                                'flow-models-hist_np=flow_models.hist_np:main',
                                'flow-models-merge=flow_models.merge:main',
                                'flow-models-plot=flow_models.plot:main',
                                'flow-models-sort=flow_models.sort:main']
        }
)
