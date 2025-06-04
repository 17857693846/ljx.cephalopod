import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
from pykrige import OrdinaryKriging

#matplotlib.rcParams['figure.figsize'] = [18, 18]
plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

def subarea_4(data):
    #0-0.1, 0.1-0.3, 0.3-0.6, 0.6-1
    m = max(data)
    q1 = max(data)*0.1
    q2 = max(data)*0.3
    q3 = max(data)*0.6
    section = []
    for i in range(len(data)):
        if data[i]>=0 and data[i]<=q1:
            section.append(10)
        elif data[i]>q1 and data[i]<=q2:
            section.append(30)
        elif data[i]>q2 and data[i]<=q3:
            section.append(60)
        elif data[i] > q3 and data[i] <= max(data):
            section.append(100)
    pies_face = [100, 60, 30, 10]
    return pies_face, section

def get_label(data):
    m = max(data)
    q1 = round(m * 0.1, 1)
    q2 = round(m * 0.3, 1)
    q3 = round(m * 0.6, 1)
    #0-q1, q1-q2, q2-q3, q3-max(m)
    str_li = []
    for i in range(4):
        if i == 0:
            str_li.append("<"+str(q1))
        elif i == 1:
            str_li.append(str(q1)+"-"+str(q2))
        elif i == 2:
            str_li.append(str(q2)+"-"+str(q3))
        elif i == 3:
            str_li.append(">"+str(q3))
    s = str_li[::-1]
    return s

# 读取CSV文件
df = pd.read_csv('predict_in_2019.csv', encoding="gbk")

# 筛选数据
months = [4, 11]
spes = [0, 1]
#fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(16, 16))
fig = plt.figure(figsize=(18, 18))
ims = []

for i, month in enumerate(months):
    for j, spe in enumerate(spes):
        # 筛选目标数据
        index = 2 * i + j + 1  # 适用于2x2的网格
        ax = fig.add_subplot(2, 2, index)
        subset = df.loc[(df["Month"] == month) & (df["Species"] == spe)]
        latitudes = subset["Lat"]
        longitudes = subset["Lon"]
        norm_values = subset["Abundance"]
        # 创建基本图层
        m = Basemap(projection='cyl', resolution='i',
                    llcrnrlat=26.5, llcrnrlon=120,
                    urcrnrlat=31.5, urcrnrlon=124.5)
        m.drawcoastlines()
        m.drawcountries(linewidth=1.5)
        m.drawstates(linewidth=0.5)
        par = np.arange(26, 33, 1)
        m.drawparallels(par, labels=[1, 0, 0, 0], fontsize=16)
        mer = np.arange(120, 126, 1)
        m.drawmeridians(mer, labels=[0, 0, 0, 1], fontsize=16)

        flo = [120, 126, 100]
        fla = [26, 32, 100]
        lon_un = np.linspace(flo[0], flo[1], flo[2])
        lat_un = np.linspace(fla[0], fla[1], fla[2])
        xgrid, ygrid = np.meshgrid(lon_un, lat_un)

        print(len(longitudes.values), len(latitudes.values), len(norm_values))
        OK = OrdinaryKriging(longitudes.values, latitudes.values, norm_values, variogram_model='spherical', nlags=6)
        zgrid, ss = OK.execute('grid', lon_un, lat_un)
        x, y = m(xgrid, ygrid)
        # 绘制克里金插值图, vmin=0, vmax=1, # levels=np.linspace(0, 1, 11),
        im = m.pcolormesh(x, y, zgrid,
                        cmap='Spectral_r', latlon=True, shading='auto', vmin=0, vmax=1)
        ims.append(im)

        m.fillcontinents(color="white", lake_color="#D7E4F1")
        print(f'{month}')
        if month==4:
            season = "Spring"
        elif month==11:
            season = "Autumn"
        if spe==0:
            name = "Loliginidae"
        elif spe==1:
            name = "Sepiolidae"
        context = "2019/{}\n{}".format(season, name)
        plt.title(context, x=0.15, y=0.6)
        # 设置标题
        #plt.title('{}'.format(year))

# 设置子图之间的间距和 colorbar
plt.tight_layout()
plt.subplots_adjust(right=0.9, wspace=0.1, hspace=0.1)
cax = plt.axes([0.92, 0.2, 0.016, 0.6])
plt.title("(2019)")
plt.colorbar(ims[1], cax=cax, label="Abundance")
plt.clim(0, 1)
#cax.set_ylim([0, 1])
plt.savefig("predict.svg", bbox_inches="tight")
plt.savefig("predict.tiff", bbox_inches="tight")
# 显示图形
plt.show()
