import io
import sys
from os import path
from setuptools import setup

if sys.version_info < (3, 10):
    sys.exit("Only Python 3.10 and greater is supported")

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with io.open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with io.open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    REQUIREMENTS = [line.rstrip() for line in f]

VERSION = "0.3.4"
DESCRIPTION = "Simplified Investment & Trading Toolkit"

KEYWORDS = [
    "Finance",
    "Toolkit",
    "Financial",
    "Analysis",
    "Fundamental",
    "Quantitative",
    "Database",
    "Equities",
    "Currencies",
    "Economics",
    "ETFs",
    "Funds",
    "Indices",
    "Moneymarkets",
    "Commodities",
    "Futures",
    "CFDs",
    "Derivatives",
    "Trading",
    "Investing",
    "Portfolio",
    "Optimization",
    "Performance",
]

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Topic :: Office/Business :: Financial :: Investment",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    "License :: OSI Approved :: MIT License",
]

INLCUDE = [
    "bbstrader",
    "bbstrader.btengine",
    "bbstrader.core",
    "bbstrader.ibkr",
    "bbstrader.metatrader",
    "bbstrader.apps",
    "bbstrader.models",
    "bbstrader.trading",
]
EXCLUDE = ["tests", "docs"]

# Setting up
setup(
    name="bbstrader",
    version=VERSION,
    author="Bertin Balouki SIMYELI",
    url="https://github.com/bbalouki/bbstrader",
    download_url="https://pypi.org/project/bbstrader/",
    project_urls={
        "Documentation": "https://bbstrader.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/bbalouki/bbstrader",
    },
    license="The MIT License (MIT)",
    author_email="<bertin@bbstrader.com>",
    maintainer="Bertin Balouki SIMYELI",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=INLCUDE,
    install_requires=REQUIREMENTS,
    extras_require={
        "MT5": ["MetaTrader5"],
    },
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    entry_points={
        "console_scripts": [
            "bbstrader=bbstrader.__main__:main",
        ],
    },
)
