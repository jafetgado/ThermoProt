import setuptools

with open('README.rst', 'r') as f:
    readme = f.read()

setuptools.setup(name='ThermoProt',
	version='1.0a1',
	author='Japheth Gado',
	author_email='japhethgado@gmail.com',
	description='A Python package to predict the thermostability of proteins with machine-learning.',
	long_description_content_type='text/x-rst',
	long_description=readme,
	url='https://github.com/jafetgado/thermoprot',
	keywords='protein thermostability machine-learning prediction',
	packages=setuptools.find_packages(),
	include_package_data=True,
	license='GNU GPLv3',
	classifiers=[
		'Programming Language :: Python :: 3',
		'Operating System :: OS Independent',
		'Topic :: Scientific/Engineering :: Bio-Informatics'
				],
	install_requires=['numpy>=1.15.4', 'pandas>=0.23.0,<0.24.0', 'scikit-learn==0.20.1'],
	python_requires='>=3'
	  	)
	  
	  

