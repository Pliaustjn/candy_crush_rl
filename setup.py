import sys
import os
import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import platform


def find_pybind11():
    """查找pybind11的include路径"""
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        import subprocess
        print("Installing pybind11...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
        import pybind11
        return pybind11.get_include()


# 确保pybind11已安装
pybind11_include = find_pybind11()

# 获取当前目录（setup.py所在目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")

print(f"当前目录: {current_dir}")
print(f"源码目录: {src_dir}")
print(f"pybind11 include: {pybind11_include}")

# 列出src目录下的所有文件
print("\nsrc目录内容:")
if os.path.exists(src_dir):
    for f in os.listdir(src_dir):
        full_path = os.path.join(src_dir, f)
        print(f"  {full_path} (存在: {os.path.exists(full_path)})")
else:
    print(f"  src目录不存在!")

# 使用glob查找cpp文件
cpp_pattern = os.path.join(src_dir, "*.cpp")
cpp_files = glob.glob(cpp_pattern)
print(f"\n找到的cpp文件: {cpp_files}")

# 如果glob也没找到，尝试手动列出
if not cpp_files:
    print("glob没找到文件，尝试手动查找...")
    # 尝试不同的路径格式
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.cpp'):
                full_path = os.path.join(root, file)
                cpp_files.append(full_path)
                print(f"  手动找到: {full_path}")

# 确保我们有两个cpp文件
required_files = ['pybind_wrapper.cpp', 'candy_crush.cpp']
source_files = []

for req_file in required_files:
    found = False
    for cpp_file in cpp_files:
        if req_file in cpp_file:
            source_files.append(cpp_file)
            found = True
            break

    if not found:
        # 尝试直接构建路径
        direct_path = os.path.join(src_dir, req_file)
        if os.path.exists(direct_path):
            source_files.append(direct_path)
        else:
            print(f"错误: 找不到 {req_file}")
            sys.exit(1)

print(f"\n将要编译的源文件: {source_files}")

# 根据平台设置编译标志
if platform.system() == 'Windows':
    extra_compile_args = ['/std:c++17', '/O2', '/EHsc']
else:
    extra_compile_args = ['-std=c++17', '-O3']

ext_modules = [
    Extension(
        "candy_crush_cpp",
        sources=source_files,
        include_dirs=[
            pybind11_include,
            src_dir,
            current_dir
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="candy_crush_cpp",
    version="1.0.0",
    description="Candy Crush Saga Environment Simulator (C++ backend)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)