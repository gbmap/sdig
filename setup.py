from importlib.metadata import entry_points
from setuptools import setup, find_packages

setup(
    name="sdig",
    version="0.1.0",
    packages=find_packages(include=["stable_diffusion_image_generator"]),
    package_data={"stable_diffusion_image_generator": ["*.yml"]},
    include_package_data=True,
    install_requires=[
        "diffusers",
        "transformers",
        "ftfy",
        "super_resolution @ git+https://github.com/gbmap/super_resolution"
    ],
    entry_points={
        "console_scripts": [
            "sdig = stable_diffusion_image_generator.sdig:main"
        ]
    },
)
