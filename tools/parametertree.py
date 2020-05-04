# -*- coding: utf -*-
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np

app = QtGui.QApplication([])
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget

import jax.numpy as jnp
from jax import lax
from jax import vmap



## test subclassing parameters
## This parameter automatically generates two child parameters which are always reciprocals of each other
class ComplexParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = False
        pTypes.GroupParameter.__init__(self, **opts)
        
        self.addChild({'name': 'A = 1/B', 'type': 'float', 'value': 7, 'suffix': 'Hz', 'siPrefix': True})
        self.addChild({'name': 'B = 1/A', 'type': 'float', 'value': 1/7., 'suffix': 's', 'siPrefix': True})
        self.a = self.param('A = 1/B')
        self.b = self.param('B = 1/A')
        self.a.sigValueChanged.connect(self.aChanged)
        self.b.sigValueChanged.connect(self.bChanged)
        
    def aChanged(self):
        self.b.setValue(1.0 / self.a.value(), blockSignal=self.bChanged)

    def bChanged(self):
        self.a.setValue(1.0 / self.b.value(), blockSignal=self.aChanged)

        

## test add/remove
## this group includes a menu allowing the user to add new parameters into its child list
class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        opts['addList'] = ['str', 'float', 'int']
        pTypes.GroupParameter.__init__(self, **opts)
    
    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(dict(name="ScalableParam %d" % (len(self.childs)+1), type=typ, value=val, removable=True, renamable=True))

class MyParameters(pTypes.GroupParameter):
    def __init__(self, config, callback, speed, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        self.myParams = dict()
        self.callback = callback

        for name, value in config.items():
            if type(value) in [float, int]:
                self.addChild({'name': name, 'type': 'float', 'value': value})
            elif type(value) is str:
                self.addChild({'name': name, 'type': 'str', 'value': value})
            else:
                raise

            p = self.param(name)
            if speed is 'fast':
                p.sigValueChanging.connect(lambda: self._callback(name))
            elif speed is 'slow':
                p.sigValueChanged.connect(lambda : self._callback(name))
            else:
                raise
            self.myParams[name] = p
    def _callback(self, name):
        print("My Parameter changed:", name)
        self.callback({name: p.value() for name,p in self.myParams.items()})
        
    


config_cost = dict(softmax_temp=1, softmax_shift1=0, softmax_shift2=0, reg_l2_self=2/5, reg_l2_other=0, reg_ent_self=0, reg_ent_other=0)
config_stream = dict(N=32)
config_grid = dict(xlim=4, ylim=4, xcenter=0, ycenter=0)
config_traj = dict(x1=1., x2=1., learning_rate=2e-2, time_scale=1., num_iter=1e4, noise=0)
config_plot = dict(xlabel='$x_1$', ylabel='$x_2$')
config_plot = {**config_grid, **config_plot}

mw = MatplotlibWidget(size=(1,1))
subplot = mw.getFigure().add_subplot(111)
line, = subplot.plot(np.linspace(0,1,100))

J = jnp.array([[2,-1],[1,2.]])
def g(x, k):
    return J@x

def scan(g, x0, num_iter):
    return lax.scan(lambda x,k: (x-g(x,k), x),
            x0, np.arange(int(num_iter)))[1]

def update_cost(params):
    if params is not None: config_cost = params

    mw.draw()

def update_traj(params):
    if params is not None: config_traj = params

    x0 = jnp.array([config_traj['x1'], config_traj['x2']])
    lr = config_traj['learning_rate']
    data = scan(lambda *x: lr*g(*x), x0, config_traj['num_iter'])
    line.set_xdata(data[:,0])
    line.set_ydata(data[:,1])
    mw.draw()

def update_grid(params):
    if params is not None: config_grid = params

    update_cost(None)

def update_stream(params):
    if params is not None: config_stream = params
    update_cost(None)

def update_plot(params):
    if params is not None: config_plot = params
    update_cost(None)
    def lims(s):
        lim, cen = config_plot[s+'lim'], config_plot[s+'center']
        return [cen-lim/2, cen+lim/2]
    print(config_plot)
    subplot.set_xlim(lims('x'))
    subplot.set_ylim(lims('y'))
    mw.draw()

params = [
        MyParameters(name='Trajectory', speed='fast', config=config_traj, callback=update_traj),
        MyParameters(name='Cost ', speed='slow', config=config_cost, callback=update_cost),
        MyParameters(name='Grid', speed='slow', config=config_grid, callback=update_grid),
        MyParameters(name='Stream', speed='slow', config=config_stream, callback=update_stream),
        MyParameters(name='Plot', speed='fast', config=config_plot, callback=update_plot),
    {'name': 'Stackelberg', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
    #{'name': 'Gradient', 'type': 'colormap'},
    #{'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
    #{'name': 'List', 'type': 'list', 'values': [1,2,3], 'value': 2},
    #{'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3,3,3]}, 'value': 2},
    {'name': 'Reset', 'type': 'action'},
    ComplexParameter(name='Custom parameter group (reciprocal values)'),
    ScalableGroup(name="Expandable Parameter Group", children=[
        {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
        {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
    ]),
    {'name': 'Extra Parameter Options', 'type': 'group', 'children': [
        {'name': 'Read-only', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'readonly': True},
        {'name': 'Renamable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'renamable': True},
        {'name': 'Removable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'removable': True},
    ]},
]

params += [
    {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
        {'name': 'Save State', 'type': 'action'},
        {'name': 'Restore State', 'type': 'action', 'children': [
            {'name': 'Add missing items', 'type': 'bool', 'value': True},
            {'name': 'Remove extra items', 'type': 'bool', 'value': True},
        ]},
    ]},
]

## Create tree of Parameter objects
p = Parameter.create(name='params', type='group', children=params)

## If anything changes in the tree, print a message
def change(param, changes):
    
    print("tree changes:")
    for param, change, data in changes:
        path = p.childPath(param)
        if path is not None:
            childName = '.'.join(path)
        else:
            childName = param.name()
        print('  parameter: %s'% childName)
        print('  change:    %s'% change)
        print('  data:      %s'% str(data))
        print('  ----------')

        if 'softmax temperature' in childName:
            line.set_ydata(np.sin(0.5*np.linspace(0,1,100)*data))
            mw.draw()

        
    
p.sigTreeStateChanged.connect(change)


def valueChanging(param, value):

    print("Value changing (not finalized): %s %s" % (param, value))
    
# Too lazy for recursion:
for child in p.children():
    child.sigValueChanging.connect(valueChanging)
    for ch2 in child.children():
        ch2.sigValueChanging.connect(valueChanging)
        


def save():
    global state
    state = p.saveState()
    
def restore():
    global state
    add = p['Save/Restore functionality', 'Restore State', 'Add missing items']
    rem = p['Save/Restore functionality', 'Restore State', 'Remove extra items']
    p.restoreState(state, addChildren=add, removeChildren=rem)
p.param('Save/Restore functionality', 'Save State').sigActivated.connect(save)
p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(restore)


## Create two ParameterTree widgets, both accessing the same data
t = ParameterTree()
t.setParameters(p, showTop=False)
t.setWindowTitle('pyqtgraph example: Parameter Tree')
#t2 = ParameterTree()
#t2.setParameters(p, showTop=False)
mw = MatplotlibWidget(size=(1,1))
subplot = mw.getFigure().add_subplot(111)
line, = subplot.plot(np.linspace(0,1,100))

#win = QtGui.QWidget()
app = pg.QtGui.QApplication([])
win = pg.LayoutWidget()
#layout = pg.LayoutWidget()#QtGui.QGridLayout()
#win.setLayout(layout)
cols = 1
win.addWidget(QtGui.QLabel("Dynamics Simulator"), 0,  0, 1, cols)
win.addWidget(mw, 1, 0, 1, cols)
win.addWidget(t, 1, 1, 1, cols,)
win.show()
win.resize(800,800)

mw.draw()

## test save/restore
s = p.saveState()
p.restoreState(s)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
