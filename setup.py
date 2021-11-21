from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
<<<<<<< HEAD
REPO_NAME = "Transfer_learning_DLCVNL_demo"
AUTHOR_USER_NAME = "wongannnee"
=======
REPO_NAME = "Tranfer_learning_DLCVNL_demo"
AUTHOR_USER_NAME = "c17hawke"
>>>>>>> 11d9f49677d93c7e10fcadaa476c3196604cdeed
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="Trasnfer learning demo for DLCVNLP Aug",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="sunny.c17hawke@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.6",
    install_requires=LIST_OF_REQUIREMENTS
)
