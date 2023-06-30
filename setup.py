from setuptools import setup,find_packages
from typing import List

HYPHAN_E_DOT = "-e ."

def get_requirements(filepath:str)->[str]:
    requirements = []

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n","")for i in requirements]

        if HYPHAN_E_DOT in requirements:
            requirements.remove(HYPHAN_E_DOT)

        return requirements


setup(
    name="AirlineReviewsprediction",
    version="0.0.1",
    author="yash Mohite",
    author_email="mohite.yash@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)