from setuptools import setup, find_packages

setup(
    name="tmd_pimc",
    version="0.0.0",
    description="tmd_pimc package (editable install helper)",
    packages=find_packages(where="numerics"),
    package_dir={"": "numerics"},
    include_package_data=True,
)

