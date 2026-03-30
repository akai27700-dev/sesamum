from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import sys
import os

# Windows用の設定
if sys.platform == "win32":
    compile_args = [
        '/O2',  # 高速化
        '/march:native',  # ネイティブ最適化
        '/std:c++20',  # C++20
        '/openmp',  # OpenMPサポート
        '/DCPPHTTPLIB_OPENSSL_SUPPORT=0',  # SSL無効
        '/DPYBIND11_DETAILED_ERROR_MESSAGES',  # 詳細エラー
        '/EHsc',  # 例外処理
        '/bigobj',  # 大きいオブジェクトファイル
    ]
    link_args = [
        '/LTCG',  # リンク時最適化
        '/OPENMP',  # OpenMPリンク
    ]
else:
    compile_args = [
        '-O3',
        '-march=native',
        '-std=c++20',
        '-fopenmp',
        '-DPYBIND11_DETAILED_ERROR_MESSAGES'
    ]
    link_args = ['-fopenmp']

ext_modules = [
    Pybind11Extension(
        "engine.othello_engine",
        sources=[
            "engine/othello_engine.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            ".",  # カレントディレクトリ
        ],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++',
        cxx_std=20,
    ),
]

setup_kwargs = dict(
    name="engine.othello_engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)

if __name__ == "__main__":
    from setuptools import setup
    setup(**setup_kwargs)
