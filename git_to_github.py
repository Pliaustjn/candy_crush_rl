"""
Git操作自动化脚本 - 将项目提交到GitHub
替代PyCharm的GUI操作，直接在命令行完成所有Git操作
"""

import os
import subprocess
import sys
from pathlib import Path


def run_git_command(command, cwd=None, check=True):
    """
    执行Git命令并打印输出

    Args:
        command: Git命令字符串或列表
        cwd: 工作目录
        check: 是否检查返回码

    Returns:
        subprocess.CompletedProcess对象
    """
    if isinstance(command, str):
        command = command.split()

    print(f"\n{'=' * 60}")
    print(f"执行命令: {' '.join(command)}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if check and result.returncode != 0:
        print(f"命令执行失败，返回码: {result.returncode}")
        return result

    return result


def setup_git_repository(project_dir):
    """
    初始化Git仓库并设置远程

    Args:
        project_dir: 项目目录路径
    """
    print("\n" + "=" * 70)
    print("开始设置Git仓库...")
    print("=" * 70)

    # 检查目录是否存在
    if not os.path.exists(project_dir):
        print(f"错误: 项目目录不存在: {project_dir}")
        return False

    os.chdir(project_dir)
    print(f"工作目录: {os.getcwd()}")

    # 检查是否已经是Git仓库
    git_dir = os.path.join(project_dir, '.git')
    if not os.path.exists(git_dir):
        print("\n1. 初始化Git仓库...")
        result = run_git_command("git init")
        if result.returncode != 0:
            print("Git初始化失败")
            return False
    else:
        print("\nGit仓库已存在，跳过初始化")

    # 创建master分支
    print("\n2. 设置master分支...")
    result = run_git_command("git branch -M master")
    if result.returncode != 0:
        print("设置master分支失败")
        return False

    # 设置远程仓库
    print("\n3. 设置远程仓库...")

    # 先检查是否已存在远程仓库
    result = run_git_command("git remote -v", check=False)

    if "origin" in result.stdout:
        print("远程仓库origin已存在，更新URL...")
        run_git_command("git remote remove origin")

    remote_url = "https://github.com/Pliaustjn/candy_crush_rl.git"
    result = run_git_command(f"git remote add origin {remote_url}")
    if result.returncode != 0:
        print("添加远程仓库失败")
        return False

    print(f"远程仓库已设置: {remote_url}")
    return True


def create_gitignore(project_dir):
    """
    创建.gitignore文件

    Args:
        project_dir: 项目目录
    """
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
*.egg

# C++编译产物
*.o
*.obj
*.dll
*.dylib
*.pyd

# Jupyter Notebook
.ipynb_checkpoints/

# TensorBoard日志
runs/
training_logs_*/
*.tfevents.*

# 训练模型和日志
*.pth
training_log_*.txt
training_log_*.csv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# 系统文件
.DS_Store
Thumbs.db

# Colab
.colab/
"""

    gitignore_path = os.path.join(project_dir, '.gitignore')

    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print(f"\n已创建.gitignore文件: {gitignore_path}")
    else:
        print(f"\n.gitignore文件已存在，跳过创建")


def commit_and_push(project_dir,
                    commit_message="Initial commit: C++ RL environment with Gym wrapper and visualization"):
    """
    提交代码并推送到GitHub

    Args:
        project_dir: 项目目录
        commit_message: 提交信息
    """
    print("\n" + "=" * 70)
    print("提交代码到GitHub...")
    print("=" * 70)

    os.chdir(project_dir)

    # 1. 查看当前状态
    print("\n1. 查看Git状态...")
    run_git_command("git status")

    # 2. 添加所有文件
    print("\n2. 添加所有文件到暂存区...")
    result = run_git_command("git add .")
    if result.returncode != 0:
        print("添加文件失败")
        return False

    # 再次查看状态
    print("\n3. 查看暂存区状态...")
    run_git_command("git status")

    # 3. 提交代码
    print(f"\n4. 提交代码...")
    print(f"提交信息: {commit_message}")
    result = run_git_command(['git', 'commit', '-m', commit_message])

    if result.returncode != 0:
        # 检查是否因为"nothing to commit"
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            print("没有需要提交的更改")
        else:
            print("提交失败")
            return False

    # 4. 推送到GitHub
    print("\n5. 推送到GitHub master分支...")
    result = run_git_command("git push -u origin master")

    if result.returncode != 0:
        # 可能需要在GitHub上设置token
        print("\n推送失败！可能的原因：")
        print("1. 需要在GitHub上创建Personal Access Token")
        print("2. 仓库不存在或没有权限")
        print("\n解决方案：")
        print("1. 访问 https://github.com/settings/tokens")
        print("2. 创建新的token（需要repo权限）")
        print("3. 使用以下命令设置远程URL：")
        print(f"   git remote set-url origin https://YOUR_TOKEN@github.com/Pliaustjn/candy_crush_rl.git")
        print("4. 重新运行推送命令")
        return False

    print("\n" + "=" * 70)
    print("✓ 代码已成功推送到GitHub!")
    print(f"仓库地址: https://github.com/Pliaustjn/candy_crush_rl")
    print(f"分支: master")
    print("=" * 70)
    return True


def verify_project_structure(project_dir):
    """
    验证项目结构是否完整

    Args:
        project_dir: 项目目录

    Returns:
        bool: 是否所有必要文件都存在
    """
    print("\n" + "=" * 70)
    print("验证项目结构...")
    print("=" * 70)

    required_files = [
        "src/candy_crush.h",
        "src/candy_crush.cpp",
        "src/pybind_wrapper.cpp",
        "src/candy_crush_network.py",
        "src/train_mcts_cpp.py",
        "src/__init__.py",
        "generate_clean_files.py",
        "setup.py",
        "CMakeLists.txt",
        "gym_env_wrapper.py",
        "visualization.py",
        "colab_setup.ipynb",
        "colab_demo.py",
        ".gitignore"
    ]

    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(project_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - 缺失!")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n警告: 发现 {len(missing_files)} 个文件缺失")
        return False

    print(f"\n✓ 所有 {len(required_files)} 个文件都存在")
    return True


def main():
    """主函数：执行完整的Git操作流程"""

    print("=" * 70)
    print("Candy Crush RL - Git自动提交脚本")
    print("=" * 70)
    print("\n此脚本将执行以下操作：")
    print("1. 初始化Git仓库")
    print("2. 创建master分支")
    print("3. 设置GitHub远程仓库")
    print("4. 创建.gitignore文件")
    print("5. 验证项目结构")
    print("6. 提交所有文件")
    print("7. 推送到GitHub master分支")
    print("\n" + "=" * 70)

    # 获取项目目录
    # 默认为当前目录，也可以通过命令行参数指定
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        # 默认使用当前脚本所在目录的父目录（项目根目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = script_dir

    # 标准化路径
    project_dir = os.path.abspath(project_dir)
    print(f"\n项目目录: {project_dir}")

    # 确认操作
    if len(sys.argv) < 3 or sys.argv[2] != "--yes":
        response = input("\n是否继续? (y/n): ")
        if response.lower() != 'y':
            print("操作已取消")
            return

    # 步骤1: 创建.gitignore
    create_gitignore(project_dir)

    # 步骤2: 初始化Git仓库
    if not setup_git_repository(project_dir):
        print("\nGit仓库设置失败!")
        return

    # 步骤3: 验证项目结构
    verify_project_structure(project_dir)

    # 步骤4: 提交并推送
    if not commit_and_push(project_dir):
        print("\n提交推送失败!")
        return

    print("\n" + "=" * 70)
    print("所有操作完成！")
    print("=" * 70)
    print(f"\n你可以在以下地址查看代码：")
    print(f"https://github.com/Pliaustjn/candy_crush_rl")
    print(f"\n在Google Colab中使用以下命令克隆仓库：")
    print(f"!git clone -b master https://github.com/Pliaustjn/candy_crush_rl.git")


if __name__ == "__main__":
    # 执行主函数
    main()