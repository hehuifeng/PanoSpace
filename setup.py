from setuptools import setup, find_packages

setup(
    name='panospace',
    version='0.1.0',
    description='PanoSpace enables precise single-cell spatial tissue analysis.',
    author='He Hui-Feng',
    author_email='huifeng@mails.ccnu.edu.cn',
    url='https://github.com/hehuifeng/PanoSpace',
    packages=find_packages(),  
    install_requires=[  
        'numpy',
        'pandas',
    ],
    classifiers=[  
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Ubuntu 22.04.4 LTS',
    ],
    python_requires='>=3.11',  # Specify Python version compatibility
)