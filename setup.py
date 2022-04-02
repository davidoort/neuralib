from setuptools import setup, find_packages

description = ('A pure Python and NumPy implementation of a neural networks'
               'library developed to understand deep neural networks deeply.')

with open('README.md') as f:
    long_description = f.read()

setup(
    name='neuralib',
    version='0.0.1',
    author='David Alonso',
    description=description,
    long_description=long_description,
    license='MIT',
    keywords='neural-networks educational machine-learning deep-learning',
    install_requires=['numpy>=1.22.0'],
    packages=find_packages(),
    test_suite='tests',
    classifiers=[
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)