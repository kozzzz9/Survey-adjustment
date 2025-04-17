import numpy as np
import os
import math
from pathlib import Path
def read_dem_data():
    try:
        current_dir = Path(__file__).resolve().parent

        dem_dir = current_dir / "1. 数字高程模型拟合"
        if not dem_dir.is_dir():
            raise NotADirectoryError(f"路径 {dem_dir} 不是有效目录")

        target_file = dem_dir / "Known.txt"
        if not target_file.exists():
            raise FileNotFoundError(f"文件 {target_file} 不存在")


        with open(target_file, 'r', encoding='gbk') as f:
            lines = f.readlines()
            n = int(lines[0].strip())  # 已知点总数
            t = int(lines[1].strip())  # 多项式系数个数
            known_data = []
            for line in lines[2:]:
                if line.strip():
                    values = [float(x) for x in line.strip().split()]
                    known_data.append({
                        'point': int(values[0]),
                        'x': values[1],
                        'y': values[2],
                        'h': values[3]
                    })


        current_dir = Path(__file__).resolve().parent


        dem_dir = current_dir / "1. 数字高程模型拟合"
        if not dem_dir.is_dir():
            raise NotADirectoryError(f"路径 {dem_dir} 不是有效目录")

        target_file = dem_dir / "UnKnown.txt"
        if not target_file.exists():
            raise FileNotFoundError(f"文件 {target_file} 不存在")


        with open(target_file, 'r', encoding='gbk') as f:
            lines = f.readlines()
            m = int(lines[0].strip())  # 待内插点数
            unknown_data = []
            for line in lines[1:]:
                if line.strip():
                    values = [float(x) for x in line.strip().split()]
                    unknown_data.append({
                        'point': int(values[0]),
                        'x': values[1],
                        'y': values[2]
                    })

        return known_data, unknown_data, t

    except Exception as e:
        print(f"读取DEM数据时出错：{str(e)}")
        return None, None, None
def read_traverse_data():

    try:
        current_dir = Path(__file__).resolve().parent


        dem_dir = current_dir / "2. 附合导线平差"
        if not dem_dir.is_dir():
            raise NotADirectoryError(f"路径 {dem_dir} 不是有效目录")

        target_file = dem_dir / "Data.txt"
        if not target_file.exists():
            raise FileNotFoundError(f"文件 {target_file} 不存在")


        with open(target_file, 'r', encoding='gbk') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        current_dir = Path(__file__).resolve().parent


        dem_dir = current_dir / "2.附和导线平差"
        if not dem_dir.is_dir():
            raise NotADirectoryError(f"路径 {dem_dir} 不是有效目录")

        target_file = dem_dir / "Data.txt"
        if not target_file.exists():
            raise FileNotFoundError(f"文件 {target_file} 不存在")


        with open(target_file, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]
    current_line = 0


    first_line = lines[current_line].split()
    Tmode = int(first_line[0])
    if Tmode == 3:
        xA = float(first_line[1])
        yA = float(first_line[2])
        T0 = dms_to_rad(first_line[3])
    elif Tmode == 4:
        xA = float(first_line[1])
        yA = float(first_line[2])
        xB = float(first_line[3])
        yB = float(first_line[4])
        T0 = math.atan2(xB - xA, yB - yA)
    else:
        raise ValueError(f"已知方位角T0输入模式:{Tmode}没有定义!")

    current_line += 1


    second_line = lines[current_line].split()
    Pnumber = int(second_line[0])
    if Pnumber <= 2:
        raise ValueError(f"总点数(={Pnumber})太少或没有未知点,无平差问题!")

    Tmode = int(second_line[1])
    if Tmode == 3:
        xC = float(second_line[2])
        yC = float(second_line[3])
        Tn1 = dms_to_rad(second_line[4])
    elif Tmode == 4:
        xC = float(second_line[2])
        yC = float(second_line[3])
        xD = float(second_line[4])
        yD = float(second_line[5])
        Tn1 = math.atan2(xD - xC, yD - yC)
    else:
        raise ValueError(f"已知方位角Tn1输入模式:{Tmode}没有定义!")

    current_line += 1


    angle = []
    s = []
    for i in range(Pnumber - 1):
        obs = lines[current_line].split()
        tempDH = int(obs[0])
        angle.append(dms_to_rad(obs[1]))
        s.append(float(obs[2]))
        current_line += 1


    last_obs = lines[current_line].split()
    tempDH = int(last_obs[0])
    angle.append(dms_to_rad(last_obs[1]))
    current_line += 1


    precision = lines[current_line].split()
    mb = float(precision[0])
    a = float(precision[1])
    b = float(precision[2])

    return {
        'start_point': {'x': xA, 'y': yA, 'azimuth': T0},
        'end_point': {'x': xC, 'y': yC, 'azimuth': Tn1},
        'observations': [{'point': i + 1, 'angle': angle[i], 'distance': s[i] if i < len(s) else None}
                         for i in range(Pnumber)],
        'weights': {'angle': mb, 'distance_fixed': a, 'distance_scale': b}
    }
def read_gnss_data(file_path=None):
    try:

        current_dir = Path(__file__).resolve().parent

        dem_dir = current_dir / "4. GNSS向量网平差"
        if not dem_dir.is_dir():
            raise NotADirectoryError(f"路径 {dem_dir} 不是有效目录")

        target_file = dem_dir / "Data.txt"
        if not target_file.exists():
            raise FileNotFoundError(f"文件 {target_file} 不存在")

        try:
            with open(target_file, 'r', encoding='gbk') as f:
                content = f.read()
                lines = content.splitlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()

        lines = [line.strip() for line in lines if line.strip()]

        print("文件前10行:")
        for i in range(min(10, len(lines))):
            print(f"{i + 1}: {lines[i]}")

        current_line = 0

        first_line = lines[current_line].split()
        n_stations = int(first_line[0])
        n_known = int(first_line[1])
        n_groups = int(first_line[2])
        n_vectors = int(first_line[3])
        current_line += 1

        print(f"总点数: {n_stations}, 已知点数: {n_known}, 向量组数: {n_groups}, 向量总数: {n_vectors}")

        stations = []
        for i in range(n_stations):
            station_info = lines[current_line].split()
            stations.append({
                'name': station_info[0],
                'x': float(station_info[1]),
                'y': float(station_info[2]),
                'z': float(station_info[3])
            })
            current_line += 1

        all_vectors = []
        L = []

        print("站点信息:")
        for station in stations:
            print(f"{station['name']}: ({station['x']}, {station['y']}, {station['z']})")

        cov_matrix = np.zeros((n_vectors * 3, n_vectors * 3))  # 完整的协方差矩阵

        total_vectors_read = 0

        vector_group_starts = []
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) == 2:
                try:
                    group_id = int(parts[0])
                    n_vecs = int(parts[1])
                    if 1 <= group_id <= n_groups:
                        vector_group_starts.append(i)
                except ValueError:
                    pass

        print(f"找到向量组起始行: {vector_group_starts}")

        for g in range(len(vector_group_starts)):
            group_start = vector_group_starts[g]
            group_end = vector_group_starts[g + 1] if g + 1 < len(vector_group_starts) else len(lines)

            group_lines = lines[group_start:group_end]
            print(f"\n处理向量组 {g + 1} 的数据，行范围: {group_start + 1}-{group_end}")

            group_info = group_lines[0].split()
            group_id = int(group_info[0])
            n_group_vectors = int(group_info[1])
            print(f"向量组 {group_id}，包含 {n_group_vectors} 个向量")

            vectors_start = 1

            group_vectors = []
            for j in range(vectors_start, vectors_start + n_group_vectors):
                if j >= len(group_lines):
                    print(f"警告: 向量组 {group_id} 数据不完整")
                    break

                vector_info = group_lines[j].split()
                if len(vector_info) < 5:
                    print(f"警告: 向量数据格式错误: {group_lines[j]}")
                    continue

                from_station = vector_info[0]
                to_station = vector_info[1]

                from_idx = next((i for i, s in enumerate(stations) if s['name'] == from_station), -1)
                to_idx = next((i for i, s in enumerate(stations) if s['name'] == to_station), -1)

                if from_idx == -1 or to_idx == -1:
                    print(f"警告: 找不到站点: {from_station} 或 {to_station}")
                    continue

                dx = float(vector_info[2])
                dy = float(vector_info[3])
                dz = float(vector_info[4])

                print(f"向量 {from_station} -> {to_station}: ({dx}, {dy}, {dz})")

                vec_info = {
                    'group': group_id,
                    'from_station': from_station,
                    'to_station': to_station,
                    'from_idx': from_idx,
                    'to_idx': to_idx,
                    'dx': dx,
                    'dy': dy,
                    'dz': dz
                }

                all_vectors.append(vec_info)
                group_vectors.append(vec_info)
                L.extend([dx, dy, dz])

            if not group_vectors:
                print(f"警告: 向量组 {group_id} 没有有效向量，跳过")
                continue

            vectors_end = vectors_start + len(group_vectors)
            cov_start = vectors_end

            cov_lines = group_lines[cov_start:]
            print(f"协方差矩阵数据行数: {len(cov_lines)}")

            n_group_obs = len(group_vectors) * 3
            group_cov = np.zeros((n_group_obs, n_group_obs))

            row_line_indices = []
            for i, line in enumerate(cov_lines):
                parts = line.split()
                if parts and parts[0].isdigit():
                    row_num = int(parts[0])
                    if 1 <= row_num <= n_group_obs:
                        row_line_indices.append((i, row_num))

            for idx in range(len(row_line_indices)):
                line_idx, row_num = row_line_indices[idx]
                row = row_num - 1

                next_line_idx = row_line_indices[idx + 1][0] if idx + 1 < len(row_line_indices) else len(cov_lines)

                row_data = []
                for i in range(line_idx, next_line_idx):
                    parts = cov_lines[i].split()
                    if i == line_idx and parts[0].isdigit():
                        for val in parts[1:]:
                            row_data.append(val)
                    else:
                        for val in parts:
                            row_data.append(val)

                print(f"第{row_num}行数据: {row_data}")

                col = 0
                for val_str in row_data:
                    try:

                        if 'e' in val_str.lower() or 'E' in val_str:
                            try:
                                base, exp = val_str.lower().split('e')
                                val = float(base) * (10 ** float(exp))
                            except ValueError:
                                print(f"警告: 无法解析科学计数法值: {val_str}")
                                continue
                        else:
                            val = float(val_str)

                        if col <= row:
                            group_cov[row, col] = val
                        col += 1
                    except ValueError:
                        print(f"警告: 无法解析值: {val_str}")

            for r in range(n_group_obs):
                for c in range(r + 1, n_group_obs):
                    group_cov[r, c] = group_cov[c, r]

            for i in range(n_group_obs):
                if group_cov[i, i] <= 0:
                    print(f"警告: 向量组 {group_id} 协方差矩阵对角元素 ({i},{i}) 非正，设置为默认值")
                    group_cov[i, i] = 1.0e-4  # 默认值

            print(f"向量组 {group_id} 的协方差矩阵已读取，大小: {group_cov.shape}")
            print(f"协方差矩阵对角元素: {np.diag(group_cov)}")

            start_idx = total_vectors_read * 3
            end_idx = start_idx + n_group_obs
            cov_matrix[start_idx:end_idx, start_idx:end_idx] = group_cov

            total_vectors_read += len(group_vectors)

            current_line = group_start + 1

        if not all_vectors:
            raise ValueError("没有读取到有效的向量数据")

        if total_vectors_read < n_vectors:
            print(f"警告: 预期读取 {n_vectors} 个向量，实际读取 {total_vectors_read} 个")
            n_obs = total_vectors_read * 3
            cov_matrix = cov_matrix[:n_obs, :n_obs]

        print(f"总协方差矩阵尺寸: {cov_matrix.shape}")
        print(f"总协方差矩阵对角元素数量: {len(np.diag(cov_matrix))}")

        L = np.array(L)
        n = len(L)
        t = (n_stations - n_known) * 3

        initial_coordinates = np.zeros((n_stations, 3))

        for i in range(n_known):
            initial_coordinates[i, 0] = stations[i]['x']
            initial_coordinates[i, 1] = stations[i]['y']
            initial_coordinates[i, 2] = stations[i]['z']

        for i in range(n_known, n_stations):

            connecting_vectors = []
            for vec in all_vectors:
                if vec['to_idx'] == i and vec['from_idx'] < n_known:
                    connecting_vectors.append({
                        'from_idx': vec['from_idx'],
                        'dx': vec['dx'],
                        'dy': vec['dy'],
                        'dz': vec['dz']
                    })

            if connecting_vectors:

                vec = connecting_vectors[0]
                from_idx = vec['from_idx']
                initial_coordinates[i, 0] = initial_coordinates[from_idx, 0] + vec['dx']
                initial_coordinates[i, 1] = initial_coordinates[from_idx, 1] + vec['dy']
                initial_coordinates[i, 2] = initial_coordinates[from_idx, 2] + vec['dz']
                print(
                    f"计算未知点 {stations[i]['name']} 的近似坐标: ({initial_coordinates[i, 0]}, {initial_coordinates[i, 1]}, {initial_coordinates[i, 2]})")
            else:

                print(f"警告: 未找到连接到未知点 {stations[i]['name']} 的向量，使用默认坐标")

                initial_coordinates[i, 0] = np.mean([stations[j]['x'] for j in range(n_known)])
                initial_coordinates[i, 1] = np.mean([stations[j]['y'] for j in range(n_known)])
                initial_coordinates[i, 2] = np.mean([stations[j]['z'] for j in range(n_known)])

        A = np.zeros((n, t))

        try:

            if np.max(np.abs(cov_matrix)) < 1e-3:
                cov_matrix = cov_matrix * 10000  # 从m²转为cm²

            P = np.linalg.inv(cov_matrix)
            print("已成功计算权矩阵P")
        except np.linalg.LinAlgError:
            print("警告: 协方差矩阵奇异，使用伪逆计算权矩阵")
            P = np.linalg.pinv(cov_matrix)

        for i, vec in enumerate(all_vectors):
            if i * 3 + 2 >= n:
                break

            from_idx = vec['from_idx']
            to_idx = vec['to_idx']

            if from_idx >= n_known:
                A[i * 3:(i + 1) * 3, (from_idx - n_known) * 3:(from_idx - n_known + 1) * 3] = -np.eye(3)

            if to_idx >= n_known:
                A[i * 3:(i + 1) * 3, (to_idx - n_known) * 3:(to_idx - n_known + 1) * 3] = np.eye(3)

        for i in range(n_stations):
            stations[i]['x'] = initial_coordinates[i, 0]
            stations[i]['y'] = initial_coordinates[i, 1]
            stations[i]['z'] = initial_coordinates[i, 2]
        print(L)
        return {
            'n': n,
            't': t,
            'A': A,
            'L': L,
            'P': P,
            'stations': stations,
            'n_known': n_known,
            'vectors': all_vectors,
            'initial_coordinates': initial_coordinates,
            'cov_matrix': cov_matrix
        }

    except Exception as e:
        print(f"读取GNSS数据时出错：{str(e)}")
        import traceback
        traceback.print_exc()
        return None
def dem_fitting(known_data, unknown_data, t, fp):
    try:
        print("\n======  DEM拟合计算  ======", file=fp)

        print("\n已知点数据：", file=fp)
        print("点号\tX(m)\tY(m)\tH(m)", file=fp)
        for point in known_data:
            print(f"{point['point']}\t{point['x']:.3f}\t{point['y']:.3f}\t{point['h']:.3f}", file=fp)

        print("\n待内插点数据：", file=fp)
        print("点号\tX(m)\tY(m)", file=fp)
        for point in unknown_data:
            print(f"{point['point']}\t{point['x']:.3f}\t{point['y']:.3f}", file=fp)


        n = len(known_data)
        A = np.zeros((n, t))
        L = np.zeros(n)

        for i, point in enumerate(known_data):
            x, y = point['x'], point['y']
            A[i, 0] = 1
            A[i, 1] = x
            A[i, 2] = y
            if t > 3:
                A[i, 3] = x * x
                A[i, 4] = x * y
                A[i, 5] = y * y
            L[i] = point['h']


        P = np.eye(n)


        N = A.T @ P @ A
        U = A.T @ P @ L


        try:
            X = np.linalg.inv(N) @ U
        except np.linalg.LinAlgError:
            print("法方程系数阵奇异，无法求解！", file=fp)
            return None


        V = A @ X - L
        sigma0 = math.sqrt((V.T @ P @ V) / (n - t))


        Qxx = np.linalg.inv(N)
        Dxx = sigma0 ** 2 * Qxx


        print("\n拟合结果：", file=fp)
        print("\n拟合参数：", file=fp)
        param_names = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5']
        for i in range(t):
            print(f"{param_names[i]} = {X[i]:.6f}", file=fp)

        print("\n精度评定：", file=fp)
        print(f"单位权中误差：σ0 = ±{sigma0:.6f}m", file=fp)
        print("\n参数精度：", file=fp)
        for i in range(t):
            print(f"{param_names[i]}的中误差：σ{param_names[i]} = ±{math.sqrt(Dxx[i, i]):.6f}", file=fp)


        print("\n未知点高程：", file=fp)
        print("点号\tX(m)\tY(m)\tH(m)", file=fp)
        for point in unknown_data:
            x, y = point['x'], point['y']
            h = X[0] + X[1] * x + X[2] * y
            if t > 3:
                h += X[3] * x * x + X[4] * x * y + X[5] * y * y
            print(f"{point['point']}\t{x:.3f}\t{y:.3f}\t{h:.3f}", file=fp)

        return X, sigma0, Dxx

    except Exception as e:
        print(f"DEM拟合计算时出错：{str(e)}", file=fp)
        return None
def dms_to_rad(dms_str):

    try:
        if isinstance(dms_str, str) and ('°' in dms_str or "'" in dms_str):

            parts = dms_str.replace('°', ' ').replace("'", ' ').replace('"', ' ').split()
            d = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
        else:

            dms = float(dms_str)
            if dms >= 1000000:
                d = int(dms / 10000)
                m = int((dms % 10000) / 100)
                s = float(f"{dms % 100:.1f}")
            else:
                d = int(dms / 10000)
                m = int((dms % 10000) / 100)
                s = float(f"{dms % 100:.1f}")

        if not (0 <= m < 60 and 0 <= s < 60):
            raise ValueError(f"无效的分或秒：{m}分{s}秒")

        return (d + m / 60.0 + s / 3600.0) * math.pi / 180
    except Exception as e:
        raise ValueError(f"无效的角度格式：{dms_str}，错误：{str(e)}")
def rad_to_dms(rad):

    degrees = rad * 180 / math.pi
    d = int(degrees)
    m = int((degrees - d) * 60)
    s = ((degrees - d) * 60 - m) * 60
    return f"{d}°{m}'{s:.3f}\""
def parse_angle(angle_str):

    try:
        if isinstance(angle_str, str):

            angle_str = angle_str.replace(' ', '')

            if '°' in angle_str:

                parts = angle_str.replace('°', ' ').replace("'", ' ').replace('"', ' ').split()
                return {
                    'degrees': int(parts[0]),
                    'minutes': int(parts[1]),
                    'seconds': float(parts[2])
                }
            else:

                angle = float(angle_str)
                if angle >= 1000000:
                    degrees = int(angle / 10000)
                    minutes = int((angle % 10000) / 100)
                    seconds = float(f"{angle % 100:.3f}")
                else:
                    degrees = int(angle / 10000)
                    minutes = int((angle % 10000) / 100)
                    seconds = float(f"{angle % 100:.3f}")
                return {
                    'degrees': degrees,
                    'minutes': minutes,
                    'seconds': seconds
                }
        else:
            raise ValueError("输入必须是字符串")
    except Exception as e:
        raise ValueError(f"无效的角度格式：{angle_str}，错误：{str(e)}")
def format_angle(degrees, minutes, seconds):

    if degrees >= 100:
        return f"{degrees:03d}°{minutes:02d}'{seconds:06.3f}\""
    else:
        return f"{degrees:02d}°{minutes:02d}'{seconds:06.3f}\""
def cal_azimuth(x1, y1, x2, y2):

    dx = x2 - x1
    dy = y2 - y1
    azimuth = math.atan2(dx, dy)
    if azimuth < 0:
        azimuth += 2 * math.pi
    return azimuth
def traverse_adjustment(data, fp):
    try:
        import traceback

        start_point = data['start_point']
        end_point = data['end_point']
        observations = data['observations']
        weights = data['weights']


        fp.write("\n=== 原始数据 ===\n")
        fp.write("起点坐标：\n")
        fp.write(f"X = {start_point['x']:.3f}m\n")
        fp.write(f"Y = {start_point['y']:.3f}m\n")
        fp.write(f"起点方位角：{rad_to_dms(start_point['azimuth'])}\n\n")

        fp.write("终点坐标：\n")
        fp.write(f"X = {end_point['x']:.3f}m\n")
        fp.write(f"Y = {end_point['y']:.3f}m\n")
        fp.write(f"终点方位角：{rad_to_dms(end_point['azimuth'])}\n\n")

        fp.write("观测数据：\n")
        fp.write("点号\t观测角\t\t边长(cm)\n")
        for obs in observations:
            angle_str = rad_to_dms(obs['angle'])
            if obs['distance'] is not None:
                fp.write(f"{obs['point']}\t{angle_str}\t{obs['distance']*100:.3f}\n")
            else:
                fp.write(f"{obs['point']}\t{angle_str}\t-\n")

        start = data['start_point']
        end = data['end_point']
        beta_obs = np.array([obs['angle'] for obs in data['observations']])  # 角度观测值（弧度）
        dist_obs = np.array([obs['distance'] * 100 for obs in data['observations'][:-1]])  # 边长观测值，转换为厘米

        n = len(beta_obs)
        x_approx = np.zeros(n)
        y_approx = np.zeros(n)
        alpha = np.zeros(n + 1)


        x_approx[0] = start['x']
        y_approx[0] = start['y']
        alpha[0] = start['azimuth']
        alpha[1] = alpha[0] + beta_obs[0]
        

        for i in range(1, n):

            alpha[i + 1] = alpha[i] + beta_obs[i] - math.pi
            if alpha[i] < 0:
                alpha[i] += 2 * math.pi
            elif alpha[i] >= 2 * math.pi:
                alpha[i] -= 2 * math.pi


            x_approx[i] = x_approx[i - 1] + (dist_obs[i - 1] / 100) * math.cos(alpha[i])
            y_approx[i] = y_approx[i - 1] + (dist_obs[i - 1] / 100) * math.sin(alpha[i])


        init_wx = x_approx[n - 1] - end['x']
        init_wy = y_approx[n - 1] - end['y']
        init_w_alpha = (alpha[n] - end['azimuth']) * 206265  # 转换为秒
        if init_w_alpha > 180 * 3600:
            init_w_alpha -= 360 * 3600
        elif init_w_alpha < -180 * 3600:
            init_w_alpha += 360 * 3600
            

        fp.write("\n初始闭合差：\n")
        fp.write(f"方位角闭合差：{init_w_alpha:.1f}秒\n")
        fp.write(f"X坐标闭合差：{init_wx:.4f}m\n")
        fp.write(f"Y坐标闭合差：{init_wy:.4f}m\n")
        

        rho = 206265
        num_angles = len(beta_obs)
        num_dists = len(dist_obs)
        B = np.zeros((3, num_angles + num_dists))


        B[0, num_dists:num_dists+num_angles] = 1


        for i in range(num_angles):
            y_diff = end['y'] - y_approx[i]
            x_diff = end['x'] - x_approx[i]
            B[1, i+num_dists] = - y_diff * 100 / rho
            B[2, i+num_dists] = x_diff * 100 / rho
        

        for i in range(num_dists):
            B[1, i] = math.cos(alpha[i + 1])
            B[2, i] = math.sin(alpha[i + 1])


        angle_weight = 1 / (data['weights']['angle'] ** 2)
        dist_weight = []
        for d in dist_obs:
            dist_weight.append(1 / (data['weights']['distance_fixed']*100 ) ** 2)
        
        P = np.diag(dist_weight + [angle_weight] * num_angles)
        P_inv = np.linalg.inv(P)
        
        W = np.array([init_w_alpha, init_wx * 100, init_wy * 100])
        N = B @ P_inv @ B.T
        N_inv = np.linalg.inv(N)
        

        K = -N_inv @ W
        V = P_inv @ B.T @ K
        

        V_dist = V[:num_dists].reshape(-1)
        V_angle = V[num_dists:].reshape(-1)
        

        vtpv = float(V.T @ P @ V)
        mu = math.sqrt(vtpv / 3)
        

        beta_adj = beta_obs + V_angle / rho
        dist_adj = dist_obs + V_dist

        x_adj = np.zeros(n)
        y_adj = np.zeros(n)
        alpha_adj = np.zeros(n + 1)
        
        x_adj[0] = start['x']
        y_adj[0] = start['y']
        alpha_adj[0] = start['azimuth']
        alpha_adj[1] = alpha_adj[0] + beta_adj[0]
        

        for i in range(1, n):
            alpha_adj[i + 1] = alpha_adj[i] + beta_adj[i] - math.pi
            if alpha_adj[i] < 0:
                alpha_adj[i] += 2 * math.pi
            elif alpha_adj[i] >= 2 * math.pi:
                alpha_adj[i] -= 2 * math.pi
                
            x_adj[i] = x_adj[i - 1] + (dist_adj[i - 1] / 100) * math.cos(alpha_adj[i])
            y_adj[i] = y_adj[i - 1] + (dist_adj[i - 1] / 100) * math.sin(alpha_adj[i])
        

        delta_x = np.zeros(n - 1)
        delta_y = np.zeros(n - 1)
        
        for i in range(n - 1):
            delta_x[i] = x_adj[i + 1] - x_adj[i]
            delta_y[i] = y_adj[i + 1] - y_adj[i]

        coord_errors = []
        
        for point_idx in range(2, n):

            F_x = np.zeros(num_dists + num_angles)
            F_y = np.zeros(num_dists + num_angles)
            

            for i in range(min(point_idx, num_dists)-1):
                F_x[i] = math.cos(alpha_adj[i + 1])
                F_y[i] = math.sin(alpha_adj[i + 1])
            

            for i in range(point_idx-1):
                y_diff = y_adj[point_idx-1] - y_adj[i]
                x_diff = x_adj[point_idx-1] - x_adj[i]
                F_x[num_dists + i] = -y_diff *100/ rho
                F_y[num_dists + i] = x_diff*100 / rho
            print(F_x,F_y)
            inv_p_x = F_x.T @ P_inv @ F_x - F_x.T @ P_inv @ B.T @ N_inv @ B @ P_inv @ F_x
            inv_p_y = F_y.T @ P_inv @ F_y - F_y.T @ P_inv @ B.T @ N_inv @ B @ P_inv @ F_y

            m_x = mu * math.sqrt(abs(inv_p_x))
            m_y = mu * math.sqrt(abs(inv_p_y))
            
            coord_errors.append({
                'point': observations[point_idx - 1]['point'],
                'm_x': m_x,
                'm_y': m_y
            })

        fp.write("\n=== 平差结果 ===\n")
        fp.write("改正数：\n")
        fp.write("点号\t角度改正(秒)\t边长改正(cm)\n")
        for i in range(num_angles):
            point_id = observations[i]['point']
            angle_corr = float(V_angle[i])
            dist_corr = float(V_dist[i]) if i < num_dists else 0.0
            fp.write(f"{point_id}\t{angle_corr:8.3f}\t{dist_corr:8.3f}\n")
        
        fp.write("\n方位角及坐标平差值：\n")
        fp.write("点号\t方位角\t\t坐标增量X(m)\t坐标增量Y(m)\tX坐标(m)\tY坐标(m)\n")
        # 起点
        fp.write(f"起点\t{rad_to_dms(alpha_adj[0])}\t-\t-\t{x_adj[0]:8.3f}\t{y_adj[0]:8.3f}\n")
        
        for i in range(n - 1):
            point_id = observations[i]['point']
            next_point = observations[i + 1]['point'] if i + 1 < len(observations) else "终点"
            fp.write(f"{point_id}\t{rad_to_dms(alpha_adj[i+1])}\t{delta_x[i]:8.3f}\t{delta_y[i]:8.3f}\t{x_adj[i+1]:8.3f}\t{y_adj[i+1]:8.3f}\n")
        
        fp.write("\n=== 精度评定 ===\n")
        fp.write(f"VTPV：{vtpv:.2f}\n")
        fp.write(f"单位权中误差：{mu:.1f}\n")
        
        fp.write("\n各点坐标中误差：\n")
        fp.write("点号\tm_x(cm)\tm_y(cm)\n")
        for error in coord_errors:
            fp.write(f"{error['point']}\t{error['m_x']:6.1f}\t{error['m_y']:6.1f}\n")

        return {
            'x_adj': x_adj,
            'y_adj': y_adj, 
            'alpha_adj': alpha_adj,
            'V_angle': V_angle,
            'V_dist': V_dist,
            'vtpv': vtpv,
            'mu': mu,
            'coord_errors': coord_errors
        }
        
    except Exception as e:
        fp.write(f"\n发生错误: {str(e)}\n")
        traceback.print_exc(file=fp)
        return None
def gnss_adjustment(data, fp):
    try:
        print("\n======  GNSS向量网平差计算（参数平差法）  ======", file=fp)
        n = data['n']
        t = data['t']
        A = data['A']
        L = data['L']
        P = data['P']
        stations = data['stations']
        n_known = data['n_known']
        vectors = data['vectors']
        initial_coordinates = data['initial_coordinates']
        cov_matrix = data['cov_matrix']

        print(f"\n观测值个数：{n}", file=fp)
        print(f"未知数个数：{t}", file=fp)
        print(f"多余观测：{n - t}", file=fp)


        print("\n原始观测值：", file=fp)
        print("从站\t至站\tdX(m)\tdY(m)\tdZ(m)", file=fp)
        for vec in vectors:
            from_name = vec['from_station']
            to_name = vec['to_station']
            print(f"{from_name}\t{to_name}\t{vec['dx']:.6f}\t{vec['dy']:.6f}\t{vec['dz']:.6f}", file=fp)

        print("\n站点信息（初始近似坐标）：", file=fp)
        print("点名\tX(m)\tY(m)\tZ(m)\t类型", file=fp)
        for i, station in enumerate(stations):
            station_type = "已知点" if i < n_known else "未知点"
            print(f"{station['name']}\t{station['x']:.6f}\t{station['y']:.6f}\t{station['z']:.6f}\t{station_type}", file=fp)

        l = np.zeros(n)
        for i, vector in enumerate(vectors):
            if i * 3 + 2 >= n:
                break
                
            from_idx = vector['from_idx']
            to_idx = vector['to_idx']

            from_x = stations[from_idx]['x']
            from_y = stations[from_idx]['y']
            from_z = stations[from_idx]['z']
            
            to_x = stations[to_idx]['x']
            to_y = stations[to_idx]['y']
            to_z = stations[to_idx]['z']
            dx_calc = to_x - from_x
            dy_calc = to_y - from_y
            dz_calc = to_z - from_z
            l[i*3] = vector['dx'] - dx_calc
            l[i*3+1] = vector['dy'] - dy_calc
            l[i*3+2] = vector['dz'] - dz_calc
        
        print("\n观测值残差l（cm）：", file=fp)
        print("序号\tdX(cm)\tdY(cm)\tdZ(cm)", file=fp)
        for i in range(0, len(l), 3):
            if i + 2 < len(l):
                idx = i // 3
                print(f"{idx+1}\t{l[i]*100:.6f}\t{l[i+1]*100:.6f}\t{l[i+2]*100:.6f}", file=fp)

        # 2. 构建法方程
        U = A.T @ P @ l
        N = A.T @ P @ A
        print("\n法方程系数矩阵N（摘要）：", file=fp)
        for i in range(min(4, N.shape[0])):
            row = " ".join([f"{N[i,j]:.4f}" for j in range(min(4, N.shape[1]))])
            print(row, file=fp)
            
        print("\n常数项U（摘要）：", file=fp)
        for i in range(min(6, len(U))):
            print(f"{U[i]:.6f}", file=fp)

        # 3. 求解参数改正量
        try:
            N_inv = np.linalg.inv(N)
            delta_X = N_inv @ U
        except np.linalg.LinAlgError:
            print("法方程系数矩阵奇异，使用伪逆求解", file=fp)
            N_inv = np.linalg.pinv(N)
            delta_X = N_inv @ U
        print("\nN逆矩阵（摘要）：", file=fp)
        for i in range(min(3, N_inv.shape[0])):
            row = " ".join([f"{N_inv[i,j]:.4f}" for j in range(min(3, N_inv.shape[1]))])
            print(row, file=fp)
        
        print("\n参数改正量（cm）：", file=fp)
        for i in range(0, len(delta_X), 3):
            if i + 2 < len(delta_X):
                station_idx = i // 3 + n_known
                station_name = stations[station_idx]['name'] if station_idx < len(stations) else f"未知点{i//3+1}"
                dx = delta_X[i] * 100
                dy = delta_X[i+1] * 100
                dz = delta_X[i+2] * 100
                print(f"{station_name}\t{dx:.6f}\t{dy:.6f}\t{dz:.6f}", file=fp)


        V = A @ delta_X - l
        L_adjusted = L + V

        v_tpv = float(V.T @ P @ V)
        freedom = max(1, n - t)
        sigma0 = math.sqrt(v_tpv / freedom)

        print("\n观测值改正数（cm）：", file=fp)
        print("序号\tdX(cm)\tdY(cm)\tdZ(cm)", file=fp)
        for i in range(0, len(V), 3):
            if i + 2 < len(V):
                idx = i // 3
                vx = V[i] * 100
                vy = V[i+1] * 100
                vz = V[i+2] * 100
                print(f"{idx+1}\t{vx:.6f}\t{vy:.6f}\t{vz:.6f}", file=fp)

        Dxx = sigma0 ** 2 * N_inv
        for i in range(n_known, len(stations)):
            idx = (i - n_known) * 3
            if idx + 2 < len(delta_X):
                stations[i]['x'] += delta_X[idx]
                stations[i]['y'] += delta_X[idx+1]
                stations[i]['z'] += delta_X[idx+2]
        print("\n平差坐标及精度：", file=fp)
        print("点名\tX(m)/±mX(cm)\tY(m)/±mY(cm)\tZ(m)/±mZ(cm)", file=fp)
        for i in range(n_known):
            station = stations[i]
            print(f"{station['name']}\t{station['x']:.6f}/±0.000\t{station['y']:.6f}/±0.000\t{station['z']:.6f}/±0.000", file=fp)

        for i in range(n_known, len(stations)):
            idx = (i - n_known) * 3
            if idx + 2 < len(delta_X) and idx + 2 < Dxx.shape[0]:
                station = stations[i]
                
                # 计算中误差
                mx = sigma0 * math.sqrt(max(0, Dxx[idx, idx])) * 100
                my = sigma0 * math.sqrt(max(0, Dxx[idx+1, idx+1])) * 100
                mz = sigma0 * math.sqrt(max(0, Dxx[idx+2, idx+2])) * 100
                
                print(f"{station['name']}\t{station['x']:.6f}/±{mx:.4f}\t{station['y']:.6f}/±{my:.4f}\t{station['z']:.6f}/±{mz:.4f}", file=fp)
        # 输出精度评定
        print("\n=== 精度评定 ===", file=fp)
        print(f"VTPV：{v_tpv:.6f} cm²", file=fp)
        print(f"多余观测：{freedom}", file=fp)
        print(f"单位权中误差：±{sigma0:.6f} cm", file=fp)
        print("\n坐标协方差矩阵（cm²）：", file=fp)
        max_display = min(6, Dxx.shape[0])
        for i in range(max_display):
            row = " ".join([f"{Dxx[i,j]*10000:.4f}" for j in range(max_display)])  # 转换为cm²
            print(row, file=fp)
        # 计算平差后的基线向量
        print("\n平差后的基线向量：", file=fp)
        print("从站\t至站\tdX(m)\tdY(m)\tdZ(m)", file=fp)
        for vec in vectors:
            from_idx = vec['from_idx']
            to_idx = vec['to_idx']
            
            from_station = stations[from_idx]
            to_station = stations[to_idx]
            
            dx = to_station['x'] - from_station['x']
            dy = to_station['y'] - from_station['y']
            dz = to_station['z'] - from_station['z']
            
            print(f"{from_station['name']}\t{to_station['name']}\t{dx:.6f}\t{dy:.6f}\t{dz:.6f}", file=fp)

        return {
            'delta_X': delta_X,
            'sigma0': sigma0,
            'Dxx': Dxx,
            'V': V,
            'L_adjusted': L_adjusted,
            'vtpv': v_tpv,
            'stations': stations,
            'N_inv': N_inv
        }

    except Exception as e:
        print(f"GNSS网平差计算时出错：{str(e)}", file=fp)
        import traceback
        traceback.print_exc(file=fp)
        return None
def main():
    try:
        print("请选择题目：")
        print("1. 数字高程模型（DEM）拟合")
        print("2. 附合导线平差")
        print("3. GNSS向量网平差")
        method = int(input("请输入选项（1-3）："))

        if method == 1:
            known_data, unknown_data, t = read_dem_data()
            if known_data is None or unknown_data is None:
                return

            with open("result_dem.txt", 'w', encoding='utf-8') as fp:
                dem_fitting(known_data, unknown_data, t, fp)
                print("\n计算完成！结果已保存到result_dem.txt")

        elif method == 2:
            data = read_traverse_data()
            with open("result_traverse.txt", 'w', encoding='utf-8') as fp:
                traverse_adjustment(data, fp)
                print("\n计算完成！结果已保存到result_traverse.txt")

        elif method == 3:
            data = read_gnss_data()
            if data is None:
                return
            with open("result_gnss.txt", 'w', encoding='utf-8') as fp:
                gnss_adjustment(data, fp)
                print("\n计算完成！结果已保存到result_gnss.txt")
        else:
            print("错误：无效的选项！")
            return
    except Exception as e:
        print(f"错误：{str(e)}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()