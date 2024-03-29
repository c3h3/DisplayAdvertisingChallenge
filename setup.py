from setuptools import setup, find_packages

setup(
    name='hunkaggle',
    version='0.0.1dev',
    description='hunkaggle',
    author='Chia-Chi Chang',
    author_email='c3h3.tw@gmail.com',
    packages=find_packages(),
    package_data={'': ['*.coffee']},
    install_requires=["numpy",
                      "scipy",
                      "pandas",
                      "scikit-learn",
                      "pymongo"],
)
