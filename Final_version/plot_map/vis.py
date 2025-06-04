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
df = pd.read_csv('sep_autumn.csv')
big = np.log10(max(df["predict_CPUE"]))
# 筛选数据
years = [2014, 2015, 2016, 2017, 2018, 2019]
months = [11]
#fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(16, 16))
fig = plt.figure(figsize=(14, 18))
ims = []

for i, year in enumerate(years):
    #for j, month in enumerate(months):
        # 筛选目标数据
    fig.add_subplot(321+i)
    subset = df.loc[(df["Year"] == year) & (df["Month"] == months[0])]
    latitudes = subset["Lat"]
    longitudes = subset["Lon"]
    values = np.array(subset["E"])
    pcpue = np.log10(subset["predict_CPUE"]+1)

    #norm_values = (pcpue - min(pcpue)) / (max(pcpue) - min(pcpue))
    norm_values = pcpue/big
    norm_values = norm_values/max(norm_values)
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
    # 绘制散点气泡图
    #m.scatter(longitudes, latitudes, latlon=True, c=norm_values, cmap='cool', alpha=0.7, s=80)

    '''gridded_lon, gridded_lat = np.meshgrid(np.arange(longitudes.min(), longitudes.max(), 0.5),
                                           np.arange(latitudes.min(), latitudes.max(), 0.5))'''

    '''gridded_values = griddata((longitudes.values, latitudes.values), norm_values.values, (gridded_lon, gridded_lat),
                              method='cubic')'''

    OK = OrdinaryKriging(longitudes.values, latitudes.values, norm_values, variogram_model='spherical', nlags=6)
    zgrid, ss = OK.execute('grid', lon_un, lat_un)
    x, y = m(xgrid, ygrid)
    # 绘制克里金插值图, vmin=0, vmax=1, # levels=np.linspace(0, 1, 11),
    im = m.pcolormesh(x, y, zgrid,
                    cmap='Spectral_r', latlon=True, shading='auto', vmin=0, vmax=1)
    ims.append(im)
    pies_face, section = subarea_4(values)
    # x, y = Basemap(lon, lat)
    m.scatter(longitudes, latitudes, section, latlon=True, color="w", alpha=0.75, edgecolor="black")
    labels = get_label(values)
    '''pies = [int(max(origin_value) * 1), int(max(origin_value) * 0.6), int(max(origin_value) * 0.3),
     int(max(origin_value) * 0.1)]'''
    for i in range(len(labels)):
        plt.scatter([], [], c="w", alpha=0.75, s=pies_face[i], edgecolor="black", label=labels[i] + " ind./km$^2$")
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1,
               loc="lower right", prop={'size': 12}, handletextpad=-0.1)
    # plt.clabel(line, inline=True, fontsize=12)

    m.fillcontinents(color="white", lake_color="#D7E4F1")
    print(f'{year}')
    context = "{}/{}\nSepiolidae".format(year, months[0])
    plt.title(context, x=0.15, y=0.6)
    # 设置标题
    #plt.title('{}'.format(year))

# 设置子图之间的间距和 colorbar
plt.tight_layout()
'''plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.15,
                    top=0.15,
                    wspace=0.1,
                    hspace=0.12)'''

cax = plt.axes([0.9, 0.2, 0.016, 0.6])
plt.subplots_adjust(right=0.95, wspace=0, hspace=0.1, top=0.95)
plt.suptitle("(D)")
plt.colorbar(ims[1], cax=cax, label="Abundance")
plt.clim(0, 1)
#cax.set_ylim([0, 1])
plt.savefig("Fig7D.svg", bbox_inches="tight")
plt.savefig("Fig7D.tiff", bbox_inches="tight")
# 显示图形
plt.show()
