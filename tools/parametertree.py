# -*- coding: utf -*-
"""
This example demonstrates the use of pyqtgraph's parametertree system. This provides
a simple way to generate user interfaces that control sets of parameters. The example
demonstrates a variety of different parameter types (int, float, list, etc.)
as well as some customized parameter types

"""


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np

app = QtGui.QApplication([])
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget


## test subclassing parameters
## This parameter automatically generates two child parameters which are always reciprocals of each other
class ComplexParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
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

#params = [
#   {'name': 'Basic parameter data types', 'type': 'group', 'children': [
#       {'name': 'Integer', 'type': 'int', 'value': 10},
#       {'name': 'Float', 'type': 'float', 'value': 10.5, 'step': 0.1},
#       {'name': 'String', 'type': 'str', 'value': "hi"},
#       {'name': 'List', 'type': 'list', 'values': [1,2,3], 'value': 2},
#       {'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3,3,3]}, 'value': 2},
#       {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
#       {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
#       {'name': 'Gradient', 'type': 'colormap'},
#       {'name': 'Subgroup', 'type': 'group', 'children': [
#           {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
#           {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
#       ]},
#       {'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
#       {'name': 'Action Parameter', 'type': 'action'},
#   ]},
#   {'name': 'Numerical Parameter Options', 'type': 'group', 'children': [
#       {'name': 'Units + SI prefix', 'type': 'float', 'value': 1.2e-6, 'step': 1e-6, 'siPrefix': True, 'suffix': 'V'},
#       {'name': 'Limits (min=7;max=15)', 'type': 'int', 'value': 11, 'limits': (7, 15), 'default': -6},
#       {'name': 'DEC stepping', 'type': 'float', 'value': 1.2e6, 'dec': True, 'step': 1, 'siPrefix': True, 'suffix': 'Hz'},
#       
#   ]},
#   {'name': 'Extra Parameter Options', 'type': 'group', 'children': [
#       {'name': 'Read-only', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'readonly': True},
#       {'name': 'Renamable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'renamable': True},
#       {'name': 'Removable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'removable': True},
#   ]},
#   ComplexParameter(name='Custom parameter group (reciprocal values)'),
#   ScalableGroup(name="Expandable Parameter Group", children=[
#       {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
#       {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
#   ]),
#]


params = [{'name': 'Cost parameters', 'type': 'group', 'children': [
    {'name': 'softmax temperature', 'type': 'float', 'value': 2.5, 'limits': (0,10), 'step': .5},
    {'name': 'softmax shift (x_1)', 'type': 'float', 'value': 0, 'step': .5},
    {'name': 'softmax shift (x_2)', 'type': 'float', 'value': 0, 'step': .5},
    ]},
    {'name': 'Dynamics parameters', 'type': 'group', 'children': [
        {'name': 'base learning rate', 'type': 'float', 'value': 1e-3, 'step': 1e-4, 'limits':(1e-6, 1e-1)},
        {'name': 'time-scale seperation', 'type': 'float', 'value': 1, 'step': .1, 'limits':(0.01, 100)},
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

        if 'softmax' in childName:
            line.set_ydata(np.sin(0.5*np.linspace(0,1,100)*data))
            mw.draw()

        
    
p.sigTreeStateChanged.connect(change)


def valueChanging(param, value):
    
    if 'softmax' in param.name():
        line.set_ydata(np.sin(0.5*np.linspace(0,1,100)*value))
        mw.draw()
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
mw = MatplotlibWidget()
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
win.addWidget(t, 2, 0, 1, cols,)
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
