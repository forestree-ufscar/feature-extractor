import io

from setuptools import setup

with io.open("README.md", "rt", encoding="utf8") as f:
    readme = f.read()

setup(
    name="feature-extractor",
    version="1.0.0",
    url="https://github.com/forestree-ufscar/feature-extractor",
    license="Apache 2.0",
    maintainer="Rodolfo Cugler",
    maintainer_email="rodolfocugler@outlook.com",
    long_description=readme,
    packages=["feature_extractor", "descriptor_bins"],
    install_requires=[],
    include_package_data=True
)
