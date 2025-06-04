#rader charts by group
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from Data_Mining.Random_Forest.Final_version import PlotFigure as pf
import heapq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [8, 6]
plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

np.random.seed(100)
class DynamicRadarChart:
    def __init__(self, title='', legend_labels=[]):
        self.data = []
        self.title = title
        self.legend_labels = legend_labels

    def add_data(self, labels, values):
        if len(labels) != len(values):
            print("Error: The number of labels and values must match.")
            return
        self.data.append({'labels': labels, 'values': values})

    def plot_single(self, idx, title=''):
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        data = self.data[idx]
        labels = data['labels']
        values = data['values']
        N = len(labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
        if title == '':
            ax.set_title(self.title)
            filename = self.title
        else:
            ax.set_title(title)
            filename = title
        ax.grid(True)
        return fig, filename

    def plot_multiple(self, title=''):
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        N = len(self.data[0]['labels'])
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, self.data[0]['labels'])
        #ax.set_thetagrids(range(0, 1, 5), self.data[0]['labels'])
        for idx, data in enumerate(self.data):
            labels = data['labels']
            values = data['values']
            if len(values) != N:
                print(f"Error: The number of values provided for group #{idx+1} is incorrect.")
                return
            values = np.concatenate((values, [values[0]]))
            if self.legend_labels:
                ax.plot(angles, values, 'o-', linewidth=2, label=self.legend_labels[idx])
            else:
                ax.plot(angles, values, 'o-', linewidth=2, label=f'Group {idx+1}')
            ax.fill(angles, values, alpha=0.25)
            ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
        if title == '':
            ax.set_title(self.title)
            filename = self.title
        else:
            ax.set_title(title)
            filename = title
        if self.legend_labels:
            ax.legend(loc=(-0.28, 0.85))
        else:
            ax.legend(loc=(1.1, 0))
        ax.grid(True)
        return fig, filename

    def plot_multiple2(self, title=''):
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        N = len(self.data[0]['labels'])
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, self.data[0]['labels'])
        ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])

        # Define line styles, markers, and colors
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D']
        colors = ['steelblue', 'darkorange', 'r', 'c']

        for idx, data in enumerate(self.data):
            labels = data['labels']
            values = data['values']
            if len(values) != N:
                print(f"Error: The number of values provided for group #{idx + 1} is incorrect.")
                return
            values = np.concatenate((values, [values[0]]))

            # Cycle through line styles, markers, and colors
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]

            if self.legend_labels:
                ax.plot(angles, values, line_style, marker=marker, color=color, linewidth=2,
                        label=self.legend_labels[idx])
            else:
                ax.plot(angles, values, line_style, marker=marker, color=color, linewidth=2, label=f'Group {idx + 1}')

            ax.fill(angles, values, alpha=0.25, color=color)

        if title == '':
            ax.set_title(self.title)
            filename = self.title
        else:
            ax.set_title(title)
            filename = title

        ax.legend(loc=(-0.28, 0.85))
        ax.grid(True)
        return fig, filename

    def save_figure(self, fig, filename):
        plt.show()
        fig.savefig(f"{filename}.tiff", bbox_inches="tight")
        fig.savefig(f"{filename}.svg", bbox_inches="tight")
        plt.close(fig)

def read_file(feature_li, target_li, month=4):
    d4 = pd.read_csv("two_species.csv", encoding="gbk")
    d4 = d4.loc[d4["Month"] == month]
    fl = []
    for feature in feature_li:
        f = d4.iloc[:, feature]
        fl.append(f)
    if len(fl) > 1:
        flist = pd.concat(fl, axis=1)
    else:
        flist = fl[0]

    tl = []
    for target in target_li:
        t = d4.iloc[:, target]
        tl.append(t)
    if len(tl) > 1:
        tlist = pd.concat(tl, axis=1)
    else:
        tlist = tl[0]
    return flist, tlist

np.random.seed(100)
dataset = [11, 12]
ins = ["Loliginidae", "Sepiolidae"]
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "Chl-a", "SSH", "ONI", "SSS", "DO", "BT"]
feature_ins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
models = ["RF", "XGBoost", "GBDT"]
years = [2014, 2015, 2016, 2017, 2018, 2019]

def model(f, t, model="RF"):
    y = np.log(t+1)
    #X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2)
    #y=minmax_scale(t)
    if model=="RF":
        mo = RandomForestRegressor()
        par = {"n_estimators": range(65, 195, 10),
               "min_samples_split": [2],
               "min_samples_leaf": [1,3]
               }
    elif model=="XGBoost":
        mo = XGBRegressor(n_estimators=200, eta=0.02)
        par = {"n_estimators": range(10, 20, 1)}
    elif model=="GBDT":
        mo = GradientBoostingRegressor(n_estimators=200, learning_rate=0.02)
    else:
        print("now not have this model, please check")
    mo.fit(f, y)
    imp = mo.feature_importances_
    print("r2:", mo.score(f, y))
    return imp

def plot_rader_importance(spidx=0, m="RF", title=""):
    print("species name:", ins[spidx])
    f, t = read_file(feature_ins[2:], [dataset[spidx]], month=4)
    imp = model(f, t, model=m)
    f1, t1 = read_file(feature_ins[2:], [dataset[spidx]], month=11)
    imp1 = model(f1, t1, model=m)
    print(f"importance in spring by {m}:", imp)
    print(f"importance in autumn by {m}:", imp1)

    #plot rader chart
    #f'Importance of {m} for {ins[spidx]}'
    radar_chart = DynamicRadarChart(title=f"{title}\n{ins[spidx]}", legend_labels=['Spring', 'Autumn'])
    radar_chart.add_data(feature_names[2:], imp)
    radar_chart.add_data(feature_names[2:], imp1)
    fig, figname = radar_chart.plot_multiple2(title=f"{title}")
    radar_chart.save_figure(fig, figname)

if __name__=="__main__":
    plot_rader_importance(spidx=0, m="RF", title="(a)")
    plot_rader_importance(spidx=0, m="XGBoost", title="(b)")
    plot_rader_importance(spidx=0, m="GBDT", title="(c)")
    plot_rader_importance(spidx=1, m="RF", title="(d)")
    plot_rader_importance(spidx=1, m="XGBoost", title="(e)")
    plot_rader_importance(spidx=1, m="GBDT", title="(f)")












