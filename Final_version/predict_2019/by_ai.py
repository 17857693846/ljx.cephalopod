import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pykrige.ok import OrdinaryKriging
import matplotlib

plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

def subarea_4(data):
    #0-0.1, 0.1-0.3, 0.3-0.6, 0.6-1
    m = max(data)
    q1 = m * 0.1
    q2 = m * 0.3
    q3 = m * 0.6
    section = []
    for i in range(len(data)):
        if data[i]>=0 and data[i]<=q1:
            section.append(10)
        elif data[i]>q1 and data[i]<=q2:
            section.append(30)
        elif data[i]>q2 and data[i]<=q3:
            section.append(60)
        elif data[i] > q3:
            section.append(100)
    pies_face = [100, 60, 30, 10]
    return pies_face, section

def get_label(data):
    m = max(data)
    q1 = round(m * 0.1, 1)
    q2 = round(m * 0.3, 1)
    q3 = round(m * 0.6, 1)
    labels = ["<"+str(q1), str(q1)+"-"+str(q2), str(q2)+"-"+str(q3), ">"+str(q3)]
    return labels[::-1]

# 模拟读取数据
df = pd.read_csv('predict_in_2019.csv', encoding="gbk")
# 假设数据已经被正确读取到 df 中

# 定义要筛选的月份和物种
months = [4, 11]
species = [0, 1]

# 创建图形和子图
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
ims = []

# 遍历物种和月份
for i, spe in enumerate(species):
    for j, month in enumerate(months):
        # 筛选目标数据
        ax = axs[i, j]
        subset = df.loc[(df["Month"] == month) & (df["Species"] == spe)]
        latitudes = subset["Lat"]
        longitudes = subset["Lon"]
        values = subset["Abundance"]  # 假设这是要绘制的数据列

        # 创建基本图层
        m = Basemap(projection='cyl', resolution='i', llcrnrlat=26.5, llcrnrlon=120, urcrnrlat=31.5, urcrnrlon=124.5, ax=ax)
        m.drawcoastlines()
        m.drawcountries(linewidth=1.5)
        m.drawstates(linewidth=0.5)

        # 绘制散点气泡图
        x, y = m(longitudes.values, latitudes.values)
        pies_face, section = subarea_4(values)
        m.scatter(x, y, s=section, color="w", alpha=0.75, edgecolor="black")

        # 添加图例
        labels = get_label(values)
        for k in range(len(labels)):
            ax.scatter([], [], c="w", alpha=0.75, s=pies_face[k], edgecolor="black", label=labels[k] + " ind./km$^2$")
        ax.legend(scatterpoints=1, frameon=True, labelspacing=1, loc="lower right", prop={'size': 12}, handletextpad=-0.1)

        # 设置标题
        ax.set_title(f"Species {spe}, Month {month}")

# 调整布局和添加colorbar
# 调整布局
plt.tight_layout(pad=3.0)

# 添加统一的色标
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # 位置调整
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="Spectral_r"), cax=cbar_ax)
cbar.set_label('Abundance')

plt.savefig("predict.pdf", bbox_inches="tight")
plt.savefig("predict.tiff", bbox_inches="tight")

# 显示图形
plt.show()

