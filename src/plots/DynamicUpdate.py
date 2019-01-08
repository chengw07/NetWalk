"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()

plt.ion()
class DynamicUpdate():
    #Suppose we know the x range
    min_x = -1
    max_x = 4

    def __init__(self):
        self.on_launch()

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots(figsize=(12, 12))
        self.lines, = self.ax.bar([0],[0],align='center')#plot([],[], 'o')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()

       # self.figure.hold(True)
        self.xdata = []
        self.ydata = []

        self.labels = ['0', '1', '2', '3(anomaly injected)', '4']
        self.ax.set_xticklabels(self.labels)
        self.ax.set_xlabel('Time Snapshot')
        self.ax.set_ylabel('Anomaly Score')
        # self.lines.set_ydata(ydata)

        self.ax.plot([-1, 4], [0.04, 0.04], 'r--', label='threshold', linewidth=2)
        self.ax.legend(loc="upper left")




    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)

        r = np.arange(len(xdata))
        # if ydata[-1] < 0.08:
        #     self.ax.bar(r, ydata, facecolor='yellowgreen', edgecolor='white', align='center')  # set_xdata(xdata)
        # else:
        #     self.ax.bar(r, ydata, facecolor='red', edgecolor='white', align='center')  # set_xdata(xdata)

        colorA = ['yellowgreen', 'yellowgreen', 'red', 'yellowgreen']
        self.ax.bar(r, ydata, color=colorA[0:len(r)],  align='center')


        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


    #Example
    def addPoint(self, x, y):
        self.xdata.append(x)
        self.ydata.append(y)
        self.on_running(self.xdata, self.ydata)
