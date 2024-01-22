import setuptools

setuptools.setup(
    name="slime_volleyball",
    version="v0.0.1",
    description="Port of David Ha's Slime volleyball for gymnasium.",
    author="Chase McDonald",
    author_email="chasemcd@andrew.cmu.edu",
    packages=["slime_volleyball"],  # same as name
    install_requires=[
        "numpy",
        "gymnasium",
    ],  # external packages as dependencies
)
