from setuptools import setup


setup(
    name="prep_model",
    version="1.0",
    install_requires=["Click"],
    entry_points="""
        [console_scripts]
        model=prep_model:cli
        """
)
