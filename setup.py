
from setuptools import setup

with open("./README.md", 'r') as f:
    long_description = f.read()

setup(
   name='pyspellchecker',
   version='0.1.0',
   description='A trainable and minimalistic spell checker',
   license="GNU GENERAL PUBLIC LICENSE Version 3",
   long_description=long_description,
   author='David Buchaca Prats',
   author_email='david.buchaca.prats@gmail.com',
   url="https://github.com/davidbp/pyspellchecker",
   packages=['pyspellchecker'],
   install_requires=['nltk', 'editdistance', 'spacy'],
   extras_require={ 'test': ['pytest']}
)
