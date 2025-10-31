import numpy as np
from scipy.linalg import inv
from collections import defaultdict


def parse_extxyz(extxyz_path):
    """
    手动解析extxyz文件，返回构型列表
    
    每个构型是一个字典，包含：
    - 'elements': 元素符号列表（如['Rh', 'N']）
    - 'energy': 总能量（eV）
    """
    structures = []
    with open(extxyz_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # 读取并清洗空行
    
    i = 0
    while i < len(lines):
        # 1. 读取原子数（每个构型的第一行）
        try:
            num_atoms = int(lines[i])
        except ValueError:
            raise ValueError(f"第{i+1}行应为原子数，却读到：{lines[i]}")
        i += 1
        
        # 2. 读取注释行（包含能量等信息）
        if i >= len(lines):
            raise EOFError("文件结束，缺少注释行")
        comment_line = lines[i]
        i += 1
        
        # 3. 提取总能量（从注释行中找'energy=xxx'）
        energy_str = None
        for item in comment_line.split():
            if item.startswith('energy='):
                energy_str = item.split('=')[1]
                break
        if not energy_str:
            raise KeyError(f"注释行中未找到'energy'字段：{comment_line}")
        try:
            total_energy = float(energy_str)
        except ValueError:
            raise ValueError(f"能量格式错误：{energy_str}")
        
        # 4. 读取原子信息（提取元素符号）
        elements = []
        for _ in range(num_atoms):
            if i >= len(lines):
                raise EOFError("文件结束，缺少原子数据")
            atom_line = lines[i]
            # 原子行格式：元素 坐标x 坐标y 坐标z ...（取第一个字段作为元素）
            elements.append(atom_line.split()[0])
            i += 1
        
        # 5. 保存当前构型
        structures.append({
            'elements': elements,
            'energy': total_energy
        })
    
    return structures


def calculate_atom_ref(extxyz_path):
    """从extxyz文件计算原子参考能量（不依赖ASE）"""
    print(f"读取extxyz文件: {extxyz_path}")
    structures = parse_extxyz(extxyz_path)  # 用手动解析替代ASE
    n_structures = len(structures)
    if n_structures == 0:
        raise ValueError("未从extxyz文件中读取到任何构型")

    # 收集所有元素并建立映射
    all_elements = set()
    for struct in structures:
        all_elements.update(struct['elements'])
    all_elements = sorted(all_elements)
    n_elements = len(all_elements)
    element_to_idx = {elem: idx for idx, elem in enumerate(all_elements)}
    print(f"数据集中包含的元素 ({n_elements}种): {all_elements}")

    # 构建成分矩阵A和总能量矩阵E_total
    A = np.zeros((n_structures, n_elements))
    E_total = np.zeros((n_structures, 1))

    for i, struct in enumerate(structures):
        # 解析当前构型的总能量和原子数
        total_energy = struct['energy']  # 总能量（eV）
        elements = struct['elements']
        total_atoms = len(elements)
        if total_atoms == 0:
            print(f"警告：第{i}个构型不含原子，已跳过")
            continue

        # 计算每个元素的原子分数
        elem_counts = defaultdict(int)
        for symbol in elements:
            elem_counts[symbol] += 1
        # 填充成分矩阵A
        for elem, count in elem_counts.items():
            idx = element_to_idx[elem]
            A[i, idx] = count / total_atoms  # 原子分数 = 原子数 / 总原子数

        # 计算每原子总能量并填充E_total
        E_total[i, 0] = total_energy / total_atoms  # 单位：eV/atom

    # 求解E_atomRef（正规方程）
    print("开始计算原子参考能量...")
    A_T = A.T
    A_T_A = np.dot(A_T, A)
    try:
        A_T_A_inv = inv(A_T_A)
    except np.linalg.LinAlgError:
        raise RuntimeError("矩阵A^T·A不可逆，可能是因为：\n1. 构型数量少于元素种类\n2. 元素完全共现")
    
    A_T_E = np.dot(A_T, E_total)
    E_atomRef = np.dot(A_T_A_inv, A_T_E)

    # 整理结果
    atom_ref_dict = {elem: float(E_atomRef[idx]) for elem, idx in element_to_idx.items()}
    return atom_ref_dict


if __name__ == "__main__":
    extxyz_file = "structure.extxyz"  # 替换为你的文件路径
    atom_ref = calculate_atom_ref(extxyz_file)
    print("\n===== 原子参考能量 E_atomRef (单位：eV/atom) =====")
    for elem, energy in sorted(atom_ref.items()):
        print(f"{elem}: {energy:.6f}")
