import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "1.0.0"

ext_modules = [
    Pybind11Extension(
        "metatrader_client",
        [
            "src/metatrader.cpp",
        ],
        define_macros=[
            ("VERSION_INFO", __version__),
        ],
        include_dirs=[
            os.path.abspath("./include"),
        ],
        cxx_std=20,
    ),
]

setup(
    name="metatrader_client",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.12",
)

# if __name__ == "__main__":
#     setup()
