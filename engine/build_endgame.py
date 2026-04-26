import os
import subprocess
import sys


def _env_flag(name, default='1'):
    value = os.environ.get(name, default)
    return str(value).strip().lower() not in {'0', 'false', 'off', 'no'}


def build_endgame_solver():
    """Build the endgame solver with pybind11"""
    is_windows = sys.platform == 'win32'
    use_openmp = _env_flag('SESAMUM_USE_OPENMP', '1')
    if is_windows:
        compiler = 'cl'
        includes = ['/I' + os.path.dirname(os.path.abspath(__file__)), '/I' + os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), '/EHsc', '/O2']
        if use_openmp:
            includes.append('/openmp')
        defines = ['/DNOMINMAX', '/DWIN32_LEAN_AND_MEAN']
    else:
        compiler = 'g++'
        includes = ['-I' + os.path.dirname(os.path.abspath(__file__)), '-I' + os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), '-O3', '-std=c++17']
        if use_openmp:
            includes.append('-fopenmp')
        defines = ['-DNOMINMAX']
    sources = ['endgame_solver.cpp', 'othello_core_cpp.cpp', 'othello_engine.cpp', 'othello_engine_bindings.cpp']
    if is_windows:
        output = 'othello_engine.cp313-win_amd64.pyd'
    else:
        output = 'othello_engine.so'
    cmd = [compiler] + includes + defines + sources + ['/Fe:' + output] if is_windows else [compiler] + includes + defines + sources + ['-o', output]
    print(f'Building endgame solver... OpenMP={"ON" if use_openmp else "OFF"}')
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print('Build successful!')
        if result.stdout:
            print('STDOUT:', result.stdout)
    except subprocess.CalledProcessError as e:
        print(f'Build failed with error: {e}')
        print('STDERR:', e.stderr)
        return False
    return True
if __name__ == '__main__':
    if build_endgame_solver():
        print('Endgame solver built successfully!')
    else:
        print('Failed to build endgame solver.')
        sys.exit(1)
