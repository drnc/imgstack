from setuptools import setup

setup(
    name='imgstack',
    version='1.0',
    description='Stack multiple TIFF files, producing a sigma clipped average image',
    url='https://github.com/drnc/imgstack',
    author='Guillaume Duranceau',
    author_email='g2lm@drnc.net',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
    keywords='stack TIFF sigma clipped average',
    py_modules=['imgstack/imgstack'],
    install_requires=['numpy>=1.14', 'tifffile>=2018.11.6'],
    python_requires='>=3',
    entry_points={
        'console_scripts': [ 'imgstack=imgstack.imgstack:main' ],
    }
)
