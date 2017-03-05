import argparse

#CMD Arguments
description = ('Plot parsed logs')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--trainlogpath',
                    help='Path to training log.')
parser.add_argument('--testlogpath',
                    help='Path to testing log.')
parser.add_argument('-x', nargs='+', type=str,
                    help='Channel(s) to be used as x axis [Max: 2]')
parser.add_argument('-y', nargs='+', type=str,
                    help='Channel(s) to be used as y axis [Max: 2]')
parser.add_argument('--outpath',
                    help='Path to save the plot.')

args = parser.parse_args()
#args = parser.parse_args(['--trainlogpath=D:\Users\Amogh\Projects\PL2Workspace\PIRESPP\Cheekbones\log.train.average', '--testlogpath=D:\Users\Amogh\Projects\PL2Workspace\PIRESPP\Cheekbones\log.test','-x','NumIters','NumIters','-y','loss', 'accuracy'])
#args = parser.parse_args(['log.train.average','log.test','-x', 'NumIters','-y','accuracy'])

if (not args.trainlogpath) and (not args.testlogpath):
    print "[ERROR: Must provide at least one log file to plot (train or test).]"
    quit()

if args.x is None:
    args.x = [None]

if len(args.x) > 2 or len(args.y) > 2:
    print "[ERROR: Maximum two channels in x and y axes allowed.]"
    quit()

import os
import ntpath
import asciitable
import numpy as np
from matplotlib import pylab
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

def formatterFunc(x, pos):
    """ This function returns a string with 2 decimal places, given the input x"""
    return '%.2f' % x

def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s: 
            return i
    return -1

def find_first(strs,substr): 
    try: 
        out = next(x for x in strs if substr in x)
    except: 
        out = None
    return out

#main
#parameters
#trainls = '-';#testls = '-.'
#laxisc = 'b';#raxisc = 'r';
trainc = 'b'; testc = 'r'
laxisls = '-'; raxisls = '--'
lmarker = 'v'; rmarker = '^'
cushion = 1 #times std

if args.trainlogpath: 
    trainlog = asciitable.read(args.trainlogpath)

if args.testlogpath: 
    testlog = asciitable.read(args.testlogpath)

#plt.figure(figsize=(10, 7))
host = host_subplot(111)

if len(args.x) == 2: ay1 = host.twiny()
if len(args.y) == 2: ax1 = host.twinx()

host.set_xlabel(args.x[0])
host.set_ylabel(args.y[0] + " [ " + laxisls + " ]")
try: ax1.set_ylabel(args.y[1] + " [ " + raxisls + " ]")
except: pass
try: ay1.set_xlabel(args.x[1])
except: pass

trainx= args.x[:]; testx = args.x[:]
trainy= args.y[:]; testy = args.y[:]

for i in range(len(args.x)): 
    if args.trainlogpath: trainx[i] = find_first(trainlog.dtype.fields.keys(),args.x[i])
    if args.testlogpath: testx[i] = find_first(testlog.dtype.fields.keys(),args.x[i])
         
for i in range(len(args.y)): 
    if args.trainlogpath: trainy[i] = find_first(trainlog.dtype.fields.keys(),args.y[i])
    if args.testlogpath: testy[i] = find_first(testlog.dtype.fields.keys(),args.y[i])
   

trainapex = np.zeros(len(trainy))

testapex = np.zeros(len(testy))
trainapexidx = np.zeros(len(trainy), dtype = np.int)
testapex = np.zeros(len(testy))
testapexidx = np.zeros(len(trainy), dtype = np.int)

for i in range(len(args.y)): 
    if "loss" in args.y[i] and args.trainlogpath: 
        trainapex[i] = trainlog[trainy[i]].min()
        trainapexidx[i] = trainlog[trainy[i]].argmin()
    elif args.trainlogpath: 
        trainapex[i] = trainlog[trainy[i]].max()
        trainapexidx[i] = trainlog[trainy[i]].argmax()
    if "loss" in args.y[i] and args.testlogpath: 
        testapex[i] = testlog[testy[i]].min()
        testapexidx[i] = testlog[testy[i]].argmin()
    elif args.testlogpath: 
        testapex[i] = testlog[testy[i]].max()
        testapexidx[i] = testlog[testy[i]].argmax()

lowlim = []; uplim = []
if args.trainlogpath:
    if trainx[0] is not None:
        trainplot, = host.plot(trainlog[trainx[0]], trainlog[trainy[0]], 
                        label = "train: " + trainy[0] + " [" + "{:1.3f}".format(trainapex[0]) + "]", 
                        color = trainc, linestyle = laxisls)
        host.scatter(trainlog[trainx[0]][trainapexidx[0]], trainapex[0], 
                     marker = lmarker, color = trainc, alpha = 0.25, s = 50)

    else:
        trainplot, = host.plot(trainlog[trainy[0]], 
                        label = "train: " + trainy[0] + " [" + "{:1.3f}".format(trainapex[0]) + "]", 
                        color = trainc, linestyle = laxisls)
        host.scatter(trainapexidx[0], trainapex[0], 
                     marker = lmarker, color = trainc, alpha = 0.25, s = 50)

    lowlim.append( max(trainlog[trainy[0]].min() - trainlog[trainy[0]].std()*cushion*0.25, trainlog[trainy[0]].mean() - trainlog[trainy[0]].std()*cushion) )
    uplim.append( min(trainlog[trainy[0]].max() + trainlog[trainy[0]].std()*cushion*0.25, trainlog[trainy[0]].mean() + trainlog[trainy[0]].std())*cushion)
    if "accuracy" in trainy[0]: host.set_ylim( -0.01, 1.01 )
    #else: host.set_ylim( -1, 56 )
    #else: host.set_ylim( min(lowlim), max(uplim) )
if args.testlogpath:
    if testx[0] is not None:
        testplot, = host.plot(testlog[testx[0]], testlog[testy[0]], 
                        label = "test: " + testy[0] + " [" + "{:1.3f}".format(testapex[0]) + "]", 
                        color=testc, linestyle=laxisls)
        host.scatter(testlog[testx[0]][testapexidx[0]], testapex[0], 
                     marker = lmarker, color = testc, alpha = 0.25, s = 50)
    else:
        testplot, = host.plot(testlog[testy[0]], 
                    label = "test: " + testy[0] + " [" + "{:1.3f}".format(testapex[0]) + "]", 
                    color=testc, linestyle=laxisls)
        host.scatter(testapexidx[0], testapex[0], 
                     marker = lmarker, color = testc, alpha = 0.25, s = 50)
    
    lowlim.append( max(testlog[testy[0]].min() - trainlog[trainy[0]].std()*cushion*0.25, testlog[testy[0]].mean() - testlog[testy[0]].std()*cushion) )
    uplim.append( min(testlog[testy[0]].max() + trainlog[trainy[0]].std()*cushion*0.25, testlog[testy[0]].mean() + testlog[testy[0]].std()*cushion) )
    if "accuracy" in testy[0]: host.set_ylim( -0.01, 1.01 )
    #else: host.set_ylim( -1, 56 )
    #else: host.set_ylim( min(lowlim), max(uplim) )

try:
    lowlim = []; uplim = []
    if args.trainlogpath:
        if trainx[0] is not None:
            trainplot1, = ax1.plot(trainlog[trainx[0]], trainlog[trainy[1]], 
                            label = "train: " + trainy[1] + " [" + "{:1.3f}".format(trainapex[1]) + "]",  
                            color = trainc, linestyle = raxisls)
            ax1.scatter(trainlog[trainx[0]][trainapexidx[1]], trainapex[1], 
                     marker = rmarker, color = trainc, alpha = 0.25, s = 50)
        else:
            trainplot1, = ax1.plot(trainlog[trainy[1]], 
                            label = "train: " + trainy[1] + " [" + "{:1.3f}".format(trainapex[1]) + "]",  
                            color = trainc, linestyle = raxisls)
            ax1.scatter(trainapexidx[1], trainapex[1], 
                     marker = rmarker, color = trainc, alpha = 0.25, s = 50)
        lowlim.append( max(trainlog[trainy[1]].min() - trainlog[trainy[1]].std()*cushion*0.1, trainlog[trainy[1]].mean() - trainlog[trainy[1]].std()*cushion) )
        uplim.append( min(trainlog[trainy[1]].max() + trainlog[trainy[1]].std()*cushion*0.1, trainlog[trainy[1]].mean() + trainlog[trainy[1]].std())*cushion)
        if "accuracy" in trainy[1]: ax1.set_ylim( -0.01, 1.01 )
        #else: ax1.set_ylim( -1, 56 )
        #else: ax1.set_ylim( min(lowlim), max(uplim) )
    if args.testlogpath:
        if testx[0] is not None:
            testplot1, = ax1.plot(testlog[testx[0]], testlog[testy[1]], 
                            label = "test: " + testy[1] + " [" + "{:1.3f}".format(testapex[1]) + "]",
                            color=testc, linestyle=raxisls)
            ax1.scatter(testlog[testx[0]][testapexidx[1]], testapex[1], 
                     marker = rmarker, color = testc, alpha = 0.25, s = 50)
        else:
            testplot1, = ax1.plot(testlog[testy[1]], 
                            label = "test: " + testy[1] + " [" + "{:1.3f}".format(testapex[1]) + "]",
                            color=testc, linestyle=raxisls)
            ax1.scatter(testapexidx[1], testapex[1], 
                     marker = rmarker, color = testc, alpha = 0.25, s = 50)
        lowlim.append( max(testlog[testy[1]].min() - trainlog[trainy[1]].std()*cushion*0.1, testlog[testy[1]].mean() - testlog[testy[1]].std()*cushion) )
        uplim.append( min(testlog[testy[1]].max() + trainlog[trainy[1]].std()*cushion*0.1, testlog[testy[1]].mean() + testlog[testy[1]].std()*cushion) )
        if "accuracy" in testy[1]: ax1.set_ylim( -0.01, 1.01 )
        #else: ax1.set_ylim( -1, 56 )
        #else: ax1.set_ylim( min(lowlim), max(uplim) )
except: pass
try:
    if args.trainlogpath:
        ay1.set_xticks(host.get_xticks())
    if trainx[1] is not None:
        ay1.set_xticklabels( np.around(trainlog[trainx[1]][::int(host.get_xticks()[1] - host.get_xticks()[0])], decimals=1))
    else:
        ay1.set_xticklables(host.get_xticks())
except: pass

host.set_xlim(left = 0)
ax1.set_xlim(left = 0)

host.legend(loc='best', fancybox=True, framealpha=0.5) 
#plt.ylim([0,5])
plt.title(ntpath.basename(os.path.dirname(os.path.abspath(args.trainlogpath if args.trainlogpath else args.testlogpath))), y=1.08)

#host.axis["left"].label.set_color(trainplot.get_color())
#try: ax1.axis["right"].label.set_color(trainplot1.get_color())
#except: pass

plt.grid() 
plt.tight_layout()

if args.outpath: 
    plt.savefig(args.outpath)
    print "[ Plotted to", args.outpath,"]"
else: 
    print "[ Plotted to screen ]"
    plt.show()


