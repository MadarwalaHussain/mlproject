from setuptools import setup, find_packages
HYPEN_E_DOT = '-e .'
# This script is used to set up the Python package for the ML project.
def get_requirements(file_path):
    """
    This function reads the requirements from a file and returns them as a list.
    """
    requirements = []
    # Check if the file exists
    with open(file_path, 'r') as file:
        requirements = file.readlines()

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    # Remove any whitespace characters like `\n` at the end of each line
    return [req.strip() for req in requirements if req.strip()]

# This is the setup script for the ML project.
setup(
    name='mlproject',
    version='0.0.1',
    author='Hussain Madarawala',
    author_email='hussainmadar4@gmail.com',
    description='A simple ML project template',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )