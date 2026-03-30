from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import sys
import os
from setuptools import setup
import subprocess
import importlib
from typing import Any


def _load_attr(module_name: str, attr_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, attr_name, None)


def _get_msvc_helper_module() -> Any:
    return importlib.import_module("setuptools._distutils.compilers.C.msvc")


def _get_msvc_compiler_class() -> Any:
    return _load_attr("setuptools._distutils.compilers.C.msvc", "Compiler")

class CustomBuildExt(build_ext):
    """カスタム build_ext: 最適化フラグを削除"""

    @staticmethod
    def _activate_x64_msvc_env():
        msvc = _get_msvc_helper_module()
        if msvc is None:
            return

        vcvarsall, _ = msvc._find_vcvarsall("x64")
        if not vcvarsall:
            return

        cmd = f'cmd /u /c ""{vcvarsall}" x64 && set"'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-16le", errors="replace")
        env = {}
        for line in out.splitlines():
            key, _, value = line.partition("=")
            if key and value:
                env[key] = value
        os.environ.update(env)
        os.environ["DISTUTILS_USE_SDK"] = "1"
        os.environ["MSSdk"] = "1"
    
    def run(self):
        # build_extensions の前にフラグ処理をオーバーライド
        import sys
        if sys.platform == "win32":
            self._activate_x64_msvc_env()
            # distu tilsまたは setuptools の compiler モジュルをモンキーパッチ
            MSVCCompiler = _get_msvc_compiler_class()
            
            if MSVCCompiler:
                orig_spawn = MSVCCompiler.spawn
                orig_initialize = MSVCCompiler.initialize

                def patched_initialize(self, plat_name=None):
                    orig_initialize(self, plat_name)
                    self.compile_options = [arg for arg in self.compile_options if arg not in ['/O2', '/GL', '/DNDEBUG']]
                    self.compile_options_debug = [arg for arg in self.compile_options_debug if arg not in ['/GL']]
                    self.ldflags_shared = [arg for arg in self.ldflags_shared if arg != '/LTCG']
                    self.ldflags_shared_debug = [arg for arg in self.ldflags_shared_debug if arg != '/LTCG']
                    self.ldflags_exe = [arg for arg in self.ldflags_exe if arg != '/LTCG']
                    self.ldflags_exe_debug = [arg for arg in self.ldflags_exe_debug if arg != '/LTCG']
                    self.ldflags_static = [arg for arg in self.ldflags_static if arg != '/LTCG']
                    self.ldflags_static_debug = [arg for arg in self.ldflags_static_debug if arg != '/LTCG']
                
                def patched_spawn(self, cmd):
                    # コマンドラインから /O2 と /GL を削除
                    if isinstance(cmd, list):
                        cmd = [arg for arg in cmd if arg not in ['/O2', '/GL', '/DNDEBUG', '/LTCG']]
                        # /Od を追加（最初にない場合）
                        if not any(arg in ['/Od', '/O1', '/Ox'] for arg in cmd):
                            # /EHsc 後に /Od を挿入
                            for i, arg in enumerate(cmd):
                                if '/EHsc' in arg:
                                    cmd.insert(i + 1, '/Od')
                                    break
                    return orig_spawn(self, cmd)
                
                MSVCCompiler.initialize = patched_initialize
                MSVCCompiler.spawn = patched_spawn
        
        return super().run()
             
if sys.platform == "win32":
    compile_args = [
        '/std:c++20',
        '/openmp',
        '/DCPPHTTPLIB_OPENSSL_SUPPORT=0',
        '/DPYBIND11_DETAILED_ERROR_MESSAGES',
        '/EHsc',
        '/bigobj',
        '/utf-8',
        '/O2',          # Maximum optimization
        '/Ot',          # Optimize for speed
        '/Oi',          # Enable intrinsics
        '/Ob2',         # Inline expansion level 2
        '/arch:AVX2',   # Enable AVX2 instructions
        '/fp:fast',     # Fast floating point
        '/GL-',         # Disable link-time codegen for faster build
    ]
    link_args = ['/INCREMENTAL:NO', '/LTCG:OFF']
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
            "engine/othello_engine_bindings.cpp",
            "engine/othello_engine_session.cpp",
            "engine/othello_core_cpp.cpp",  # Added for fast evaluation
        ],
        include_dirs=[
            pybind11.get_include(),
            ".",              
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
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.8",
)

if __name__ == "__main__":
    from setuptools import setup
    setup(**setup_kwargs)
