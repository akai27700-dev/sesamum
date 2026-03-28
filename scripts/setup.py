from setuptools import Extension, setup

import pybind11
import platform

def get_compile_args():
    system = platform.system()
    if system == "Windows":
        return [
            "/O2", "/Ot", "/Oi", "/fp:fast", 
            "/arch:AVX2", "/favor:INTEL64", "/Zm300", "/std:c++17",
            "/D__AVX2__", "/openmp"
        ]
    else:
        return [
            "-O3", "-ffast-math", "-march=native", "-std=c++17",
            "-mavx2", "-mfma", "-fopenmp"
        ]

def get_link_args():
    system = platform.system()
    if system == "Windows":
        return ["/openmp"]
    else:
        return ["-fopenmp"]

ext_modules = [
    Extension(
        "othello_engine",
        ["engine/othello_engine.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args(),
    )
]

setup(
    name="othello_engine",
    version="0.1.0",
    ext_modules=ext_modules,
)
