import os
from pathlib import Path
from typing import List, Union

from setuptools import find_packages, setup


def _load_requirements(path: Union[str, Path], filename: str, comment_char: str = "#") -> List[str]:
    with open(os.path.join(path, filename)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        if ln.startswith("http") or "@http" in ln:
            continue
        if ln:
            reqs.append(ln)
    return reqs


PROJECT_ROOT_PATH = Path(__file__).parent
LONG_DESCRIPTION = (PROJECT_ROOT_PATH / "README.md").read_text()

main_packages = _load_requirements(PROJECT_ROOT_PATH, "requirements.txt")
dev_packages = _load_requirements(PROJECT_ROOT_PATH, "requirements-dev.txt")
demo_packages = _load_requirements(PROJECT_ROOT_PATH, "requirements-demo.txt")
extras = {"dev": dev_packages, "all": dev_packages + demo_packages}

setup(
    name="sidewalk-widths-extractor",
    version="0.1.0",
    description="TODO",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Berk Bilir & Egemen Kopuz",
    author_email="egemen.kopuz@tum.de",
    url="https://github.com/idp-sidewalk-widths-extraction/sidewalk-widths-extractor",
    license="LICENSE",
    packages=find_packages(exclude=["tests*", "demo*", "scripts*", "configs*", "docs*"]),
    python_requires=">=3.7",
    install_requires=main_packages,
    extras_require=extras,
)
