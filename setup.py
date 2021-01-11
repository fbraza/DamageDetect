from setuptools import setup


setup(
    name="yolo-edp",
    version="1.0",
    install_requires=["Click"],
    entry_points="""
        [console_scripts]
        model=app:cli
        """
)
