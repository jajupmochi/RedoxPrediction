from setuptools import setup

with open('requirements.txt') as fp:
	install_requires = fp.read()

setup(
	name='redox_prediction',
	version='1.0.0',
	packages=['redox_prediction'],
	url='https://github.com/jajupmochi/RedoxPrediction',
	license='',
	author='linlin',
	author_email='linlin.jia@unibe.ch',
	description='The codes for the RedoxPrediction project.',
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Operating System :: OS Independent",
		'Intended Audience :: Science/Research',
		'Intended Audience :: Developers',
	],
	# install_requires=install_requires, # todo
)
