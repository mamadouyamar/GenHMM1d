# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:43:42 2020

@author: 49009427
"""
import setuptools

# setuptools.setup(
#     # Needed to silence warnings (and to be a worthwhile package)
#     name='Measurements',
#     url='https://github.com/jladan/package_demo',
#     author='John Ladan',
#     author_email='jladan@uwaterloo.ca',
    
#     # Needed for dependencies
#     install_requires=['scipy','numpy','functools','joblib','multiprocessing','math','scipy.optimize','scipy.stats'],
    
#     # *strongly* suggested for sharing
#     version='0.1',
    
#     # The license can be anything you like
#     license='MIT',
#     description='An example of a python package from pre-existing code',
#     # We will also need a readme eventually (there will be a warning)
#     # long_description=open('README.txt').read(),
# )


setuptools.setup(
    name="GenHMM1d", # Replace with your own username
    version="0.0.1",
    author="Mamadou Yamar Thioub and Bouchra R Nasri",
    author_email="mamadou-yamar.thioub@hec.ca",
    description="Inference, goodness-of-fit tests, and predictions for continuous and discrete univariate Hidden Markov Models (HMM). The goodness-of-fit test is based on a Cramer-von Mises statistic and uses parametric bootstrap to estimate the p-value. The description of the methodology is taken from Nasri et al (2020) <doi: 10.1029/2019WR025122>.",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy','numpy','functools','joblib','multiprocessing','math','scipy.optimize','scipy.stats'],
    python_requires='>=3.6',
)