# zip_overlay_copy.py
# 功能：选择一个zip文件或一个文件夹 -> 选择目标文件夹 -> 将源内容“新增/覆盖”复制到目标目录（不删除）
# 说明：
# - 选择zip：会解压到临时目录后再覆盖复制
# - 只覆盖/新增：目标里多出来的文件不会被删除
# - 默认保留时间戳（copy2）

import os
import shutil
import zipfile
import tempfile
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox


def _is_within_directory(base_dir: str, target_path: str) -> bool:
    """确保 target_path 在 base_dir 下，避免 zip slip."""
    base_dir = os.path.abspath(base_dir)
    target_path = os.path.abspath(target_path)
    return os.path.commonpath([base_dir]) == os.path.commonpath([base_dir, target_path])


def safe_extract_zip(zip_path: str, extract_to: str) -> None:
    """安全解压：防止路径穿越（Zip Slip）"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = os.path.join(extract_to, member.filename)
            if not _is_within_directory(extract_to, member_path):
                raise RuntimeError(f"Unsafe path detected in zip: {member.filename}")
        zf.extractall(extract_to)


def overlay_copy_dir(src_dir: str, dst_dir: str) -> tuple[int, int, list[str]]:
    """
    将 src_dir 下所有文件/目录新增或覆盖到 dst_dir。
    返回：(复制文件数, 创建目录数, 覆盖文件列表)
    """
    copied_files = 0
    created_dirs = 0
    overwritten = []

    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        rel_root = "" if rel_root == "." else rel_root

        target_root = os.path.join(dst_dir, rel_root)
        if not os.path.exists(target_root):
            os.makedirs(target_root, exist_ok=True)
            created_dirs += 1

        # 创建目录（不删除目标目录里已有的）
        for d in dirs:
            td = os.path.join(target_root, d)
            if not os.path.exists(td):
                os.makedirs(td, exist_ok=True)
                created_dirs += 1

        # 复制文件（覆盖同名）
        for f in files:
            s = os.path.join(root, f)
            t = os.path.join(target_root, f)

            if os.path.exists(t):
                overwritten.append(os.path.relpath(t, dst_dir))

            # 确保父目录存在
            os.makedirs(os.path.dirname(t), exist_ok=True)

            # 覆盖复制
            shutil.copy2(s, t)
            copied_files += 1

    return copied_files, created_dirs, overwritten


def plan_overlay(src_dir: str, dst_dir: str) -> tuple[int, int, int]:
    """
    预演一次覆盖复制会发生什么（不做实际复制）。
    返回：(将处理的文件总数, 其中“新增文件”数, 其中“覆盖文件”数)
    """
    total = 0
    new_files = 0
    overwrite_files = 0

    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    for root, _, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        rel_root = "" if rel_root == "." else rel_root
        target_root = os.path.join(dst_dir, rel_root)

        for f in files:
            total += 1
            t = os.path.join(target_root, f)
            if os.path.exists(t):
                overwrite_files += 1
            else:
                new_files += 1

    return total, new_files, overwrite_files


def pick_source() -> tuple[str | None, str | None]:
    """
    先问用户：选zip还是选文件夹。
    返回 (path, kind) kind in {"zip","dir"} or (None,None)
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    choice = messagebox.askyesno(
        "选择源类型",
        "点【是】选择 ZIP 文件；点【否】选择一个源文件夹。",
        parent=root,
    )

    if choice:  # ZIP
        path = filedialog.askopenfilename(
            title="选择一个 ZIP 文件",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
            parent=root,
        )
        root.destroy()
        return (path if path else None, "zip" if path else None)
    else:  # DIR
        path = filedialog.askdirectory(title="选择一个源文件夹", parent=root)
        root.destroy()
        return (path if path else None, "dir" if path else None)


def pick_target_dir() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title="选择目标文件夹（会被新增/覆盖）", parent=root)
    root.destroy()
    return path if path else None


def main():
    start_ts = datetime.now()

    src_path, kind = pick_source()
    if not src_path or not kind:
        print("未选择源，退出。")
        return

    # 打印源信息（便于确认自己选的是哪一个 zip / 文件夹）
    print("=== 本次源 ===")
    print(f"源路径：{os.path.abspath(src_path)}")
    print(f"源名称：{os.path.basename(src_path)}")
    try:
        print(f"脚本文件：{os.path.basename(__file__)}")
    except Exception:
        pass

    dst_dir = pick_target_dir()
    if not dst_dir:
        print("未选择目标文件夹，退出。")
        return

    if kind == "dir":
        src_dir = src_path
        if not os.path.isdir(src_dir):
            raise RuntimeError(f"源不是文件夹：{src_dir}")

        total, new_files, overwrite_files = plan_overlay(src_dir, dst_dir)
        if total > 0 and (new_files / total) > 0.5:
            # 防止选错目标目录导致“大量新增文件”覆盖到错误项目里
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            ok = messagebox.askyesno(
                "二次确认（防止选错目录）",
                "检测到本次复制里，“新路径文件”占比超过 50%（这些文件在目标目录里原先不存在）。\n\n"
                f"源：{os.path.basename(src_dir)}\n"
                f"目标：{os.path.abspath(dst_dir)}\n\n"
                f"将处理文件总数：{total}\n"
                f"新增文件：{new_files}\n"
                f"覆盖文件：{overwrite_files}\n\n"
                "如果你是选错了目标仓库目录，点【否】退出重新选择；\n"
                "如果你确认无误，点【是】继续执行。",
                parent=root,
            )
            root.destroy()
            if not ok:
                print("用户取消：新路径文件占比过高，已退出。")
                return

        copied, created, overwritten = overlay_copy_dir(src_dir, dst_dir)

    else:  # zip
        zip_path = src_path
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(f"不是有效的 ZIP 文件：{zip_path}")

        with tempfile.TemporaryDirectory(prefix="zip_overlay_") as tmp:
            safe_extract_zip(zip_path, tmp)

            total, new_files, overwrite_files = plan_overlay(tmp, dst_dir)
            if total > 0 and (new_files / total) > 0.5:
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                ok = messagebox.askyesno(
                    "二次确认（防止选错目录）",
                    "检测到本次复制里，“新路径文件”占比超过 50%（这些文件在目标目录里原先不存在）。\n\n"
                    f"源：{os.path.basename(zip_path)}\n"
                    f"目标：{os.path.abspath(dst_dir)}\n\n"
                    f"将处理文件总数：{total}\n"
                    f"新增文件：{new_files}\n"
                    f"覆盖文件：{overwrite_files}\n\n"
                    "如果你是选错了目标仓库目录，点【否】退出重新选择；\n"
                    "如果你确认无误，点【是】继续执行。",
                    parent=root,
                )
                root.destroy()
                if not ok:
                    print("用户取消：新路径文件占比过高，已退出。")
                    return

            copied, created, overwritten = overlay_copy_dir(tmp, dst_dir)

    # 输出结果
    print("=== 完成 ===")
    print(f"目标目录：{os.path.abspath(dst_dir)}")
    print(f"创建目录数：{created}")
    print(f"复制/覆盖文件数：{copied}")
    if overwritten:
        print(f"\n覆盖的文件（最多显示前200个，共 {len(overwritten)} 个）：")
        for p in overwritten[:200]:
            print(" -", p)
        if len(overwritten) > 200:
            print(" ...（省略）")

    # 弹窗提示
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "完成",
            f"已完成新增/覆盖复制。\n\n创建目录：{created}\n复制/覆盖文件：{copied}\n覆盖文件数：{len(overwritten)}",
            parent=root,
        )
        root.destroy()
    except Exception:
        pass

    end_ts = datetime.now()
    print("\n=== 结束信息 ===")
    print(f"执行开始：{start_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"执行结束：{end_ts.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
