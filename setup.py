from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='spotify-data-pipeline',
    version='1.0.0',
    author='Akhilesh Jadhav',
    author_email='your.email@example.com',
    description='A comprehensive data pipeline for analyzing Spotify song popularity using AWS services',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/spotify-data-pipeline',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'spotify-pipeline=src.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['config/*.yaml'],
    },
)
