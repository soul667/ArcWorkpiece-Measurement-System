import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体文件路径
font_path = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'

# 创建 FontProperties 对象，指定字体路径
prop = font_manager.FontProperties(fname=font_path)

# 设置 matplotlib 使用该字体
plt.rcParams['font.sans-serif'] = [prop.get_name()]  # 使用通过 FontProperties 获取的字体名称
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 绘制图表
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('使用 Noto Serif CJK SC 字体的图表', fontproperties=prop)
plt.xlabel('横坐标', fontproperties=prop)
plt.ylabel('纵坐标', fontproperties=prop)

# 显示图表
plt.show()
