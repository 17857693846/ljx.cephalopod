import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige import OrdinaryKriging
from scipy.interpolate import Rbf
from scipy import interpolate
from mpl_toolkits.basemap import Basemap
from math import radians, sin, cos, asin, sqrt
from scipy.interpolate import griddata
import matplotlib.patches as mpatches
from matplotlib.pyplot import scatter
plt.rcParams["font.size"] = 12
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

def idw():
    pass

def plot_map(lon, lat, value, find_lon_inf=[120, 125, 300], find_lat_inf=[26, 32, 300], title="", subtime="stime",
             label="Value", iflog=False, figure_name="figure_name"):
    m = Basemap(projection='merc', llcrnrlat=27, urcrnrlat=32, llcrnrlon=120, urcrnrlon=125, resolution="i")
    m.drawmapboundary()
    m.drawcountries()
    m.drawcoastlines()
    m.drawstates()
    m.drawrivers()
    par = np.arange(26, 32, 1)
    m.drawparallels(par, labels=[1, 0, 0, 0], fontsize=10)
    mer = np.arange(120, 125, 1)
    m.drawmeridians(mer, labels=[0, 0, 0, 1], fontsize=10)

    flo = find_lon_inf
    fla = find_lat_inf
    lon_un = np.linspace(flo[0], flo[1], flo[2])
    lat_un = np.linspace(fla[0], fla[1], fla[2])
    xgrid, ygrid = np.meshgrid(lon_un, lat_un)
    #x, y = xgrid.flatten(), ygrid.flatten()
    # method={'linear', 'nearest', 'cubic'}
    '''lstz = griddata(np.column_stack([lon, lat]), value, (x, y), method="linear")
    zgrid = np.array(lstz).reshape(xgrid.shape)
    lons, lats = m(lon, lat)'''
    '''克里金插值法linear, power, gaussian, (hao)spherical, (跟idw一样)exponential, hole-effect'''
    OK = OrdinaryKriging(lon, lat, value, variogram_model='spherical', nlags=6)
    zgrid, ss = OK.execute('grid', lon_un, lat_un)

    print("max", zgrid.max())
    print("min", zgrid.min())
    if iflog == True:

        im2 = m.pcolormesh(xgrid, ygrid, np.log(zgrid + 1), cmap='Spectral_r', latlon=True, shading='auto')
        cb = m.colorbar(im2, location="right", label="ln(" + label + ")")
        m.contour(xgrid, ygrid, np.log(zgrid + 1), colors='w')

        # plt.clabel(line, inline=True, fontsize=12)

    else:

        im2 = m.pcolormesh(xgrid, ygrid, zgrid, cmap='Spectral_r', latlon=True, shading='auto')
        cb = m.colorbar(im2, location="right", label=label)
        m.contour(xgrid, ygrid, zgrid, colors='w')
        m.fillcontinents(color="white", lake_color="#D7E4F1")
        # plt.clabel(line, inline=True, fontsize=12)
    plt.title(title)
    plt.suptitle(subtime, x=0.62, y=0.2)
    plt.savefig(figure_name)
    plt.show()
    plt.cla()

def plot_map_compare(lon, lat, value, origin_value, find_lon_inf=[120, 126, 300], find_lat_inf=[26, 32, 300], title="title", subtime="stime",
             label="Value", iflog=False, figure_name="figure_name"):
    m = Basemap(projection='merc', llcrnrlat=27, urcrnrlat=32, llcrnrlon=120, urcrnrlon=125, resolution="i")
    m.drawmapboundary()
    m.drawcountries()
    m.drawcoastlines()
    m.drawstates()
    m.drawrivers()
    par = np.arange(26, 32, 1)
    m.drawparallels(par, labels=[1, 0, 0, 0], fontsize=10)
    mer = np.arange(120, 125, 1)
    m.drawmeridians(mer, labels=[0, 0, 0, 1], fontsize=10)

    flo = find_lon_inf
    fla = find_lat_inf
    lon_un = np.linspace(flo[0], flo[1], flo[2])
    lat_un = np.linspace(fla[0], fla[1], fla[2])
    xgrid, ygrid = np.meshgrid(lon_un, lat_un)
    # x, y = xgrid.flatten(), ygrid.flatten()
    # method={'linear', 'nearest', 'cubic'}
    '''lstz = griddata(np.column_stack([lon, lat]), value, (x, y), method="linear")
    zgrid = np.array(lstz).reshape(xgrid.shape)
    lons, lats = m(lon, lat)'''
    '''克里金插值法linear, power, gaussian, (hao)spherical, (跟idw一样)exponential, hole-effect'''
    OK = OrdinaryKriging(lon, lat, value, variogram_model='spherical', nlags=6)
    zgrid, ss = OK.execute('grid', lon_un, lat_un)

    print("max", zgrid.max())
    print("min", zgrid.min())
    if iflog == True:

        im2 = m.pcolormesh(xgrid, ygrid, np.log(zgrid + 1), cmap='Spectral_r', latlon=True, shading='auto')
        cb = m.colorbar(im2, location="right", label="ln(" + label + ")")
        m.contour(xgrid, ygrid, np.log(zgrid + 1), colors='w')

        # plt.clabel(line, inline=True, fontsize=12)

    else:

        im2 = m.pcolormesh(xgrid, ygrid, zgrid, cmap='Spectral_r', latlon=True, shading='auto')
        cb = m.colorbar(im2, location="right", label=label)
        m.contour(xgrid, ygrid, zgrid, colors='w')
        m.fillcontinents(color="white", lake_color="#D7E4F1")
        #x, y = Basemap(lon, lat)
        m.scatter(lon, lat, origin_value, latlon=True, color="w", alpha=0.75, edgecolor="black")
        for i in [int(max(origin_value)*0.6), int(max(origin_value)*0.3), int(max(origin_value)*0.1), int(max(origin_value)*0.05)]:
            plt.scatter([], [], c="w", alpha=0.75, s=i, edgecolor="black", label=str(i)+" ind/km$^2$")
        plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc="lower right", prop={'size': 8})
        #plt.clabel(line, inline=True, fontsize=12)
    plt.title(title)
    plt.suptitle(subtime, x=0.62, y=0.8)
    #plt.savefig(figure_name)
    plt.show()
    plt.cla()

def plot_map_line(lon, lat, value, origin_value, find_lon_inf=[120, 126, 300], find_lat_inf=[26, 32, 300], title="title", subtime="stime",
             label="Value", iflog=False, figure_name="figure_name"):
    m = Basemap(projection='merc', llcrnrlat=27, urcrnrlat=32, llcrnrlon=120, urcrnrlon=125, resolution="i")
    m.drawmapboundary()
    m.drawcountries()
    m.drawcoastlines()
    m.drawstates()
    m.drawrivers()
    par = np.arange(26, 32, 1)
    m.drawparallels(par, labels=[1, 0, 0, 0], fontsize=10)
    mer = np.arange(120, 125, 1)
    m.drawmeridians(mer, labels=[0, 0, 0, 1], fontsize=10)

    flo = find_lon_inf
    fla = find_lat_inf
    lon_un = np.linspace(flo[0], flo[1], flo[2])
    lat_un = np.linspace(fla[0], fla[1], fla[2])
    xgrid, ygrid = np.meshgrid(lon_un, lat_un)
    # x, y = xgrid.flatten(), ygrid.flatten()
    # method={'linear', 'nearest', 'cubic'}
    '''lstz = griddata(np.column_stack([lon, lat]), value, (x, y), method="linear")
    zgrid = np.array(lstz).reshape(xgrid.shape)
    lons, lats = m(lon, lat)'''
    '''克里金插值法linear, power, gaussian, (hao)spherical, (跟idw一样)exponential, hole-effect'''
    OK = OrdinaryKriging(lon, lat, value, variogram_model='spherical', nlags=6)
    zgrid, ss = OK.execute('grid', lon_un, lat_un)

    print("max", zgrid.max())
    print("min", zgrid.min())
    if iflog == True:

        im2 = m.pcolormesh(xgrid, ygrid, np.log(zgrid + 1), cmap='Spectral_r', latlon=True, shading='auto')
        cb = m.colorbar(im2, location="right", label="ln(" + label + ")")
        m.contour(xgrid, ygrid, np.log(zgrid + 1), colors='w')

        # plt.clabel(line, inline=True, fontsize=12)

    else:

        im2 = m.contourf(xgrid, ygrid, zgrid, cmap='Spectral_r', latlon=True, shading='auto')
        cb = m.colorbar(im2, location="right", label=label)
        #m.contour(xgrid, ygrid, zgrid, colors='w')

        #x, y = Basemap(lon, lat)
        m.scatter(lon, lat, origin_value, latlon=True, color="w", alpha=0.75, edgecolor="black")
        for i in [int(max(origin_value)*0.6), int(max(origin_value)*0.3), int(max(origin_value)*0.1), int(max(origin_value)*0.05)]:
            plt.scatter([], [], c="w", alpha=0.75, s=i, edgecolor="black", label=str(i)+" ind/km$^2$")
        plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc="lower right", prop={'size': 8})
        #plt.clabel(line, inline=True, fontsize=12)
        m.fillcontinents(color="white", lake_color="#D7E4F1")
    plt.title(title)
    plt.suptitle(subtime, x=0.62, y=0.8)
    #plt.savefig(figure_name)
    plt.show()
    plt.cla()

#By kriging
def plot_map_line_area(lon, lat, value, origin_value, find_lon_inf=[120, 126, 300], find_lat_inf=[26, 32, 300], title="title", subtime="stime",
             label="Value", iflog=False, figure_name="figure_name"):
    m = Basemap(projection='merc', llcrnrlat=27, urcrnrlat=32, llcrnrlon=120, urcrnrlon=125, resolution="i")
    m.drawmapboundary()
    m.drawcountries()
    m.drawcoastlines()
    m.drawstates()
    m.drawrivers()
    par = np.arange(26, 32, 1)
    m.drawparallels(par, labels=[1, 0, 0, 0], fontsize=10)
    mer = np.arange(120, 125, 1)
    m.drawmeridians(mer, labels=[0, 0, 0, 1], fontsize=10)

    flo = find_lon_inf
    fla = find_lat_inf
    lon_un = np.linspace(flo[0], flo[1], flo[2])
    lat_un = np.linspace(fla[0], fla[1], fla[2])
    xgrid, ygrid = np.meshgrid(lon_un, lat_un)
    # x, y = xgrid.flatten(), ygrid.flatten()
    # method={'linear', 'nearest', 'cubic'}
    '''lstz = griddata(np.column_stack([lon, lat]), value, (x, y), method="linear")
    zgrid = np.array(lstz).reshape(xgrid.shape)
    lons, lats = m(lon, lat)'''
    '''克里金插值法linear, power, gaussian, (hao)spherical, (跟idw一样)exponential, hole-effect(很抽象)'''
    OK = OrdinaryKriging(lon, lat, value, variogram_model='hole-effect', nlags=6)
    zgrid, ss = OK.execute('grid', lon_un, lat_un)

    print("max", zgrid.max())
    print("min", zgrid.min())
    if iflog == True:
        im2 = m.pcolormesh(xgrid, ygrid, np.log(zgrid + 1), cmap='Spectral_r', latlon=True, shading='auto')
        cb = m.colorbar(im2, location="right", label="ln(" + label + ")")
        m.contour(xgrid, ygrid, np.log(zgrid + 1), colors='w')
        # plt.clabel(line, inline=True, fontsize=12)

    else:
        im2 = m.contourf(xgrid, ygrid, zgrid, cmap='Spectral_r', latlon=True, shading='auto')
        cb = m.colorbar(im2, location="right", label=label)
        #m.contour(xgrid, ygrid, zgrid, colors='w')
        pies_face, section = subarea_4(origin_value)
        #x, y = Basemap(lon, lat)
        m.scatter(lon, lat, section, latlon=True, color="w", alpha=0.75, edgecolor="black")
        labels = get_label(origin_value)
        '''pies = [int(max(origin_value) * 1), int(max(origin_value) * 0.6), int(max(origin_value) * 0.3),
         int(max(origin_value) * 0.1)]'''
        for i in range(len(labels)):
            plt.scatter([], [], c="w", alpha=0.75, s=pies_face[i], edgecolor="black", label=labels[i]+" ind/km$^2$")
        plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc="lower right", prop={'size': 10})
        #plt.clabel(line, inline=True, fontsize=12)
        m.fillcontinents(color="white", lake_color="#D7E4F1")
    plt.title(title)
    plt.suptitle(subtime, x=0.62, y=0.8)
    plt.savefig(figure_name, bbox_inches="tight")
    plt.show()
    plt.cla()

if __name__ == "__main__":
    df = pd.read_csv("fj1.csv")
    lon = np.array(df["lon"], dtype=float)
    lat = np.array(df["lat"], dtype=float)
    value = np.array(df["value"], dtype=float)
    """find_lon_inf = [120, 126, 300]
    find_lat_inf = [26, 33, 300]"""
    plot_map_line_area(lon, lat, value, value)
    #plot_map_compare(lon, lat, value, value)

