import setuptools

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(
    name="supercells",
    version="0.1.1",
    url="https://github.com/raffaelecheula/supercells.git",

    author="Raffaele Cheula",
    author_email="cheula.raffaele@gmail.com",

    description="Tools for structure-dependent microkinetic modelling.",
    long_description=readme,
    license='GPL-3.0',

    packages=[
        'supercells',
    ],
    package_dir={
        'supercells': 'supercells'
    },
    install_requires=requirements,
    python_requires='>=3.5, <4',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
    ],
)
