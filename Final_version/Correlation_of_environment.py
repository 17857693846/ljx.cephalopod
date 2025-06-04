import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
import pandas as pd
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
#matplotlib.rcParams['figure.figsize'] = [8, 6]
plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

def display_correlation_matrix(df):
    # 计算相关系数矩阵
    corr_matrix = df.corr()

    # 设置颜色列的范围和中点
    color_range = (-1, 1)
    mid_point = 0

    # 构建颜色映射
    cmap = LinearSegmentedColormap.from_list('diverging', ['navy', 'white', 'firebrick'], N=256)

    # 创建图表
    ax = sns.heatmap(corr_matrix, cmap=cmap, center=mid_point, annot=True, fmt='.2f')

    # 绘制左下角的颜色条
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    cbar = ax.collections[0].colorbar

    # 设置颜色条标签
    cbar.ax.set_ylabel('Correlation coefficient in spring', rotation=90)

    # 设置图表标题
    plt.title('(A)')
    plt.savefig("Fig2a.svg")
    plt.savefig("Fig2a.tif")
    # 显示图表
    plt.show()

def plot_histogram_with_kde(data):
    # 计算分组范围和组数
    bin_range = (np.min(data), np.max(data))
    bin_count = 8

    # 创建直方图
    counts, bins, _ = plt.hist(data, bins=bin_count, range=bin_range, density=True, alpha=0.5, color="white", edgecolor="black")

    # 计算核概率密度函数
    kde = gaussian_kde(data)
    x_grid = np.linspace(bin_range[0], bin_range[1], 1000)
    kde_values = kde(x_grid)

    # 绘制核密度图
    plt.plot(x_grid, kde_values, 'r-')

    # 添加图表标题和轴标签
    plt.title('Histogram with KDE')
    plt.xlabel('ln(CPUE+0.01)')
    plt.ylabel('Density')

    # 显示图表
    #plt.savefig("kdeLoli.tif")
    plt.show()

    # 计算期望和方差
    expected_value = np.mean(data)
    std_dev = np.var(data)

    # 打印期望和标准差
    print("Expected value: ", expected_value)
    print("Var: ", std_dev)

if __name__=="__main__":
    d4 = pd.read_csv("two_species.csv", encoding="gbk")
    d4 = d4.loc[d4["Month"] == 4]
    #d4 = d4.loc[d4["E"] > 0]
    display_correlation_matrix(d4.iloc[:, 2:-2])
    #plot_histogram_with_kde(np.log10(d4.iloc[:, 12]+1))

