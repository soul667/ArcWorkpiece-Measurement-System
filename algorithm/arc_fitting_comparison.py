import numpy as np
import matplotlib.pyplot as plt
import time
import random
from circle_arc import CircleArc
import matplotlib.font_manager as fm


font_path = './SimSun.ttf'  
fm.fontManager.addfont(font_path)  # 动态注册字体

font_props = fm.FontProperties(fname=font_path)
font_name = font_props.get_name()  # 获取字体族名称
plt.rcParams['font.family'] = [font_name, 'SimSun']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 实验参数范围设置
ERROR_THRESHOLD = 0.5         # 误差阈值（超过该值的结果将被跳过）
RADIUS_RANGE = (3, 50)        # 半径范围(3-8mm)
CENTER_RANGE = (-50, 50)       # 圆心x,y坐标范围(-50到50mm)
START_ANGLE_RANGE = (0, 360) # 起始角度范围（度）
ANGLE_SPANS = np.linspace(7, 15, 30)  # 圆心角范围(3-5度，4个点)
REPEAT_TIMES = 1000          # 每个圆心角重复次数
NUM_POINTS = 2500           # 每个弧段的点数
NOISE_LEVEL = 0.001         # 噪声水平

def generate_random_parameters():
    """生成随机的圆参数"""
    center_x = random.uniform(CENTER_RANGE[0], CENTER_RANGE[1])
    center_y = random.uniform(CENTER_RANGE[0], CENTER_RANGE[1])
    radius = random.uniform(RADIUS_RANGE[0], RADIUS_RANGE[1])
    start_angle = random.uniform(START_ANGLE_RANGE[0], START_ANGLE_RANGE[1])
    return (center_x, center_y), radius, start_angle

def run_experiment():
    # 存储结果
    results = {
        'fit_circle_arc': {
            'center_errors': [[] for _ in ANGLE_SPANS],
            'radius_errors': [[] for _ in ANGLE_SPANS],
            'times': [[] for _ in ANGLE_SPANS],
            'centers': [[] for _ in ANGLE_SPANS],
            'radii': [[] for _ in ANGLE_SPANS],
            'true_radii': [[] for _ in ANGLE_SPANS],
            'name': 'Levenberg-Marquardt '
        },
        'hyper_circle_fit': {
            'center_errors': [[] for _ in ANGLE_SPANS],
            'radius_errors': [[] for _ in ANGLE_SPANS],
            'times': [[] for _ in ANGLE_SPANS],
            'centers': [[] for _ in ANGLE_SPANS],
            'radii': [[] for _ in ANGLE_SPANS],
            'true_radii': [[] for _ in ANGLE_SPANS],
            'name': 'Hyper Fit'
        }
    }
    
    arc = CircleArc()
    
    # 对每个圆心角进行实验
    for span_idx, angle_span in enumerate(ANGLE_SPANS):
        print(f"\n处理圆心角: {angle_span:.2f}度")
        
        # 重复实验多次
        for repeat in range(REPEAT_TIMES):
            # 生成随机参数
            center, radius, start_angle = generate_random_parameters()
            end_angle = start_angle + angle_span
            
            # 生成测试数据
            points = arc.generate_arc_points(center, radius, (start_angle, end_angle), NUM_POINTS, NOISE_LEVEL)
            
            # 测试两种方法
            for method_name in results.keys():
                method = getattr(arc, method_name)
                
                # 计时并拟合
                start_time = time.time()
                center_x, center_y, fitted_radius = method(points)
                end_time = time.time()
                
                try:
                    # 检查拟合结果是否为复数
                    if isinstance(center_x, complex) or isinstance(center_y, complex) or isinstance(fitted_radius, complex):
                        print(f"警告: {method_name} 在角度 {angle_span:.2f} 的第 {repeat} 次重复中出现复数结果，跳过该结果")
                        continue
                        
                    # 计算误差
                    center_error = float(np.sqrt((center_x - center[0])**2 + (center_y - center[1])**2))
                    radius_error = float(abs(fitted_radius - radius))
                    compute_time = float(end_time - start_time)
                    
                    # 验证结果是否为有效数值，并检查误差是否超过阈值
                    if not (np.isfinite(center_error) and np.isfinite(radius_error)):
                        print(f"警告: {method_name} 在角度 {angle_span:.2f} 的第 {repeat} 次重复中出现无效结果，跳过该结果")
                        continue
                    
                    if center_error > ERROR_THRESHOLD or radius_error > ERROR_THRESHOLD:
                        print(f"警告: {method_name} 在角度 {angle_span:.2f} 的第 {repeat} 次重复中误差超过阈值 {ERROR_THRESHOLD}，跳过该结果")
                        continue
                    
                    # 存储结果
                    results[method_name]['center_errors'][span_idx].append(center_error)
                    results[method_name]['radius_errors'][span_idx].append(radius_error)
                    results[method_name]['times'][span_idx].append(compute_time)
                    results[method_name]['centers'][span_idx].append((float(center_x), float(center_y)))
                    results[method_name]['radii'][span_idx].append(float(fitted_radius))
                    results[method_name]['true_radii'][span_idx].append(float(radius))
                except Exception as e:
                    print(f"警告: {method_name} 在角度 {angle_span:.2f} 的第 {repeat} 次重复中出现异常: {str(e)}")
                    continue
                
                if repeat % 10 == 0:
                    print(f"Method: {method_name}, Repeat: {repeat}/{REPEAT_TIMES}, "
                          f"Center Error: {center_error:.6f}, Radius Error: {radius_error:.6f}")
    
    # 创建图表（确保每个圆心角至少有一个有效结果）
    has_valid_data = False
    for method_name in results.keys():
        for errors in results[method_name]['radius_errors']:
            if len(errors) > 0:
                has_valid_data = True
                break
        if has_valid_data:
            break
    
    if not has_valid_data:
        print("错误：没有有效的拟合结果可供显示")
        return

    # 创建半径误差的箱式图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算每个方法的箱式图数据
    for method_idx, (method_name, color) in enumerate(zip(results.keys(), ['red', 'green'])):
        positions = []
        data = []
        
        for span_idx, angle_span in enumerate(ANGLE_SPANS):
            if len(results[method_name]['radius_errors'][span_idx]) > 0:
                positions.append(span_idx)
                data.append(results[method_name]['radius_errors'][span_idx])
        
        # 绘制箱式图
        bp = ax.boxplot(data, 
                       positions=[p + method_idx * 0.4 - 0.2 for p in positions],
                       widths=0.35, 
                       patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.3),
                       medianprops=dict(color='black'),
                       flierprops=dict(marker='o', markerfacecolor=color, alpha=0.5))
    
    # 设置图表属性
    ax.set_xlabel('圆心角（度）', fontsize=12)
    ax.set_ylabel('半径误差', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 设置刻度标签
    ax.set_xticks(range(len(ANGLE_SPANS)))
    ax.set_xticklabels([f"{angle:.1f}" for angle in ANGLE_SPANS], rotation=45)
    
    # 添加图例
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, alpha=0.3) 
                      for c in ['red', 'green']]
    ax.legend(legend_elements, list(results.keys()), 
             loc='upper right', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    plt.savefig('fitting_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 创建半径误差均值的散点图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name, (color, marker, linestyle) in zip(results.keys(), 
            [('red', 'o', '-'), ('green', 's', '--')]):
        # 计算每个角度的均值和标准差
        means = []
        stds = []
        for errors in results[method_name]['radius_errors']:
            if len(errors) > 0:
                means.append(np.mean(errors))
                stds.append(np.std(errors))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        means = np.array(means)
        stds = np.array(stds)
        
        # 绘制误差区域
        # fill = ax.fill_between(ANGLE_SPANS, means - stds, means + stds,
        #                color=color, alpha=0.2, linewidth=0,
        #                label=f'{method_name} 误差范围')
        
        # 绘制散点图和连线
        line = ax.plot(ANGLE_SPANS, means, 
                color=color, linestyle=linestyle,
                marker=marker, markerfacecolor='none', markeredgecolor=color,
                label=f'{results[method_name]["name"]}', markersize=6, alpha=0.8,
                markeredgewidth=1.5)[0]
        
        # 绘制误差区域边界线
        # ax.plot(ANGLE_SPANS, means + stds, color=color, linestyle=':', alpha=0.3)
        # ax.plot(ANGLE_SPANS, means - stds, color=color, linestyle=':', alpha=0.3)
    
    # 设置图表属性
    ax.set_xlabel('圆心角（度）', fontsize=12)
    ax.set_ylabel('半径误差均值', fontsize=12)
    ax.set_title('不同圆心角下的半径误差均值对比', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 调整图例
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为单独的图片文件
    plt.savefig('radius_error_means.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 导出CSV文件
    import csv
    
    with open('fitting_comparison_results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 写入表头
        csvwriter.writerow(['Method', 'Arc Angle', 'Radius Error (Mean)', 'Radius Error (Max)',
                          'Predicted Radius (Mean)', 'True Radius'])
        
        # 写入数据
        for method_name in results.keys():
            for span_idx, angle_span in enumerate(ANGLE_SPANS):
                center_errors = results[method_name]['center_errors'][span_idx]
                radius_errors = results[method_name]['radius_errors'][span_idx]
                predicted_radii = results[method_name]['radii'][span_idx]
                
                # 检查是否有有效数据
                if len(center_errors) > 0:
                    true_radii = results[method_name]['true_radii'][span_idx]
                    csvwriter.writerow([
                        method_name,
                        f"{angle_span:.2f}",
                        f"{np.mean(radius_errors):.6f}",
                        f"{np.max(radius_errors):.6f}",
                        f"{np.mean(predicted_radii):.6f}",
                        f"{np.mean(true_radii):.6f}"
                    ])
                else:
                    # 如果没有有效数据，写入NA
                    csvwriter.writerow([
                        method_name,
                        f"{angle_span:.2f}",
                        "NA", "NA", "NA", "NA"
                    ])
    
    # 打印统计结果
    print("\n=== 详细统计结果已导出到 fitting_comparison_results.csv ===")
    print("\n统计表格:")
    print("方法名称 | 圆心角 | 中心误差(均值±标准差) | 中心最大误差 | 半径误差(均值±标准差) | 半径最大误差 | 计算时间(均值±标准差) | 预测半径均值")
    print("-" * 140)
    
    for method_name in results.keys():
        for span_idx, angle_span in enumerate(ANGLE_SPANS):
            center_errors = results[method_name]['center_errors'][span_idx]
            radius_errors = results[method_name]['radius_errors'][span_idx]
            times = results[method_name]['times'][span_idx]
            predicted_radii = results[method_name]['radii'][span_idx]
            
            if len(center_errors) > 0:
                print(f"{method_name:10} | {angle_span:9.2f} | "
                      f"{np.mean(center_errors):8.6f}±{np.std(center_errors):8.6f} | "
                      f"{np.max(center_errors):12.6f} | "
                      f"{np.mean(radius_errors):8.6f}±{np.std(radius_errors):8.6f} | "
                      f"{np.max(radius_errors):12.6f} | "
                      f"{np.mean(times):8.6f}±{np.std(times):8.6f} | "
                      f"{np.mean(predicted_radii):8.6f}")
            else:
                print(f"{method_name:10} | {angle_span:9.2f} | {'NA':^38} | {'NA':^12} | {'NA':^38} | {'NA':^12} | {'NA':^38} | {'NA':^8}")
    
    print("\n=== 总体统计 ===")
    for method_name in results.keys():
        print(f"\n{method_name} 整体统计:")
        # 合并所有圆心角的数据
        all_center_errors = []
        all_radius_errors = []
        all_times = []
        for errors in results[method_name]['center_errors']:
            if len(errors) > 0:
                all_center_errors.extend(errors)
        for errors in results[method_name]['radius_errors']:
            if len(errors) > 0:
                all_radius_errors.extend(errors)
        for times in results[method_name]['times']:
            if len(times) > 0:
                all_times.extend(times)
        
        if len(all_center_errors) > 0:
            print(f"中心误差 - 均值: {np.mean(all_center_errors):.6f}, "
                  f"标准差: {np.std(all_center_errors):.6f}, "
                  f"最大值: {np.max(all_center_errors):.6f}")
            print(f"半径误差 - 均值: {np.mean(all_radius_errors):.6f}, "
                  f"标准差: {np.std(all_radius_errors):.6f}, "
                  f"最大值: {np.max(all_radius_errors):.6f}")
            print(f"计算时间 - 均值: {np.mean(all_times):.6f}, "
                  f"标准差: {np.std(all_times):.6f}, "
                  f"最大值: {np.max(all_times):.6f}")
        else:
            print("无有效数据")

if __name__ == "__main__":
    run_experiment()
