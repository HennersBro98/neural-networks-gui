"""
<Neural Network - allows easy creation and use of neural networks.>
    Copyright (C) <2020>  <Henry B>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtWidgets import QWidget,QMainWindow,QMessageBox,QHBoxLayout,QScrollArea,QMenuBar,QMenu,\
                            QAction,QLabel,QFormLayout,QFrame,QPushButton,QComboBox,QSpinBox,\
                            QDoubleSpinBox,QPlainTextEdit,QCheckBox,QSizePolicy,QFileDialog
from PyQt5.QtGui import QIcon,QPalette,QBrush,QColor
from PyQt5.QtCore import Qt
import sys
import henpy_NN as NN
import numpy as np
from numpy import array as ar
import pandas as pd
import time
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pickle
import re
import itertools
import threading

### Mainwindow object displaying GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setObjectName("MainWindow")
        self.setWindowTitle("Neural Network")
        self.resize(1800,1000)
        self.setStyleSheet("font: 12pt 'Perpetua'")
                
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.iconName = 'NN_Icon.ico'
        self.setWindowIcon(QIcon(self.iconName))
        self.horizontalLayoutSuper = QHBoxLayout(self.centralwidget)
        self.horizontalLayoutSuper.setObjectName("horizontalLayoutSuper")
        
        ### Area on which the current NN, new NN and layers sections are placed
        self.main_scrollArea = QScrollArea(self.centralwidget)
        self.main_scrollArea.setWidgetResizable(True)
        self.main_scrollArea.setObjectName("main_scrollArea")
        self.main_scrollArea_widgetContents = QWidget()
        self.main_scrollArea_widgetContents.setObjectName("main_scrollArea_widgetContents")
        self.main_scrollArea.setWidget(self.main_scrollArea_widgetContents)
        
        ### Layed out horizontally
        self.horizontalLayout = QHBoxLayout(self.main_scrollArea_widgetContents)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.setCentralWidget(self.centralwidget)
        
        ### menubar
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 516, 24))
        self.menubar.setObjectName("menubar")
        
        self.menu_File = QMenu(self.menubar)
        self.menu_File.setObjectName("menu_File")
        self.menu_File.setTitle("File")
        self.menubar.addAction(self.menu_File.menuAction())
        
        self.menu_Help = QMenu(self.menubar)
        self.menu_Help.setObjectName("menu_Help")
        self.menu_Help.setTitle("Help")
        self.menubar.addAction(self.menu_Help.menuAction())
        
        self.setMenuBar(self.menubar)
        
        self.menu_File_SaveAs = QAction(self)
        self.menu_File_SaveAs.setObjectName("menu_File_SaveAs")
        self.menu_File_SaveAs.setText("Save Neural Network As")
        self.menu_File.addAction(self.menu_File_SaveAs)
        self.menu_File_SaveAs.triggered.connect(lambda:self.saveNN(As=True))
        
        self.menu_File_Save = QAction(self)
        self.menu_File_Save.setObjectName("menu_File_Save")
        self.menu_File_Save.setText("Save Neural Network")
        self.menu_File_Save.setShortcut("Ctrl+S")
        self.menu_File.addAction(self.menu_File_Save)
        self.menu_File_Save.triggered.connect(self.saveNN)
        
        self.menu_File_UploadNN = QAction(self)
        self.menu_File_UploadNN.setObjectName("menu_File_UploadNN")
        self.menu_File_UploadNN.setText("Upload Pre-Existing Neural Network")
        self.menu_File_UploadNN.setShortcut("Ctrl+U")
        self.menu_File.addAction(self.menu_File_UploadNN)
        self.menu_File_UploadNN.triggered.connect(self.uploadNN)
        
        self.menu_Help_Instructions = QAction(self)
        self.menu_Help_Instructions.setObjectName("menu_Help_Instructions")
        self.menu_Help_Instructions.setText("Instructions")
        self.menu_Help_Instructions.setShortcut("Ctrl+I")
        self.menu_Help.addAction(self.menu_Help_Instructions)
        self.menu_Help.triggered.connect(self.viewInstructions)
        ### menubar
        
        self.NN_savedName = None
        self.NN_savedFullName = None
        
        self.palette = QPalette()
        brush = QBrush(QColor('#666666'))
        brush.setStyle(Qt.SolidPattern)
        self.palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        
        ### Current NN
        self.NN_scrollArea = QScrollArea(self.main_scrollArea_widgetContents)
        self.NN_scrollArea.setWidgetResizable(True)
        self.NN_scrollArea.setObjectName("NN_scrollArea")
        self.NN_scrollArea.setFixedWidth(399)
        self.NN_scrollArea_widgetContents = QWidget()
        self.NN_scrollArea_widgetContents.setObjectName("NN_scrollArea_widgetContents")
        self.NN_scrollArea.setWidget(self.NN_scrollArea_widgetContents)
        self.NN_formLayout = QFormLayout(self.NN_scrollArea_widgetContents)
        self.NN_formLayout.setObjectName("NN_formLayout")
        
        NN_pos = 0        
        self.NN_title_top = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_title_top.setObjectName("NN_title_top")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_title_top)
        self.NN_title_top.setAlignment(Qt.AlignCenter)
        self.NN_title_top.setMaximumHeight(40)
        self.NN_title_top.setStyleSheet("font: bold 12pt")
        self.NN_title_top.setText("Current Neural Network")
        
        NN_pos += 1
        self.NN_line_top = QFrame(self.NN_scrollArea_widgetContents)
        self.NN_line_top.setFrameShape(QFrame.HLine)
        self.NN_line_top.setFrameShadow(QFrame.Plain)
        self.NN_line_top.setLineWidth(1)
        self.NN_line_top.setObjectName("NN_line_top")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_line_top)
        self.NN_line_top.setPalette(self.palette)
        
        NN_pos += 1     
        self.NN_title_dataPre = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_title_dataPre.setObjectName("NN_title_dataPre")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_title_dataPre)
        self.NN_title_dataPre.setAlignment(Qt.AlignCenter)
        self.NN_title_dataPre.setMaximumHeight(40)
        self.NN_title_dataPre.setStyleSheet("font: bold 11pt")
        self.NN_title_dataPre.setText("Data Preprocessing")

        NN_pos += 1     
        self.NN_lab_featureScale = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_featureScale.setObjectName("NN_lab_featureScale")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.LabelRole, self.NN_lab_featureScale)
        self.NN_lab_featureScale.setMaximumHeight(40)
        self.NN_lab_featureScale.setText("Feature scaling:")
        self.widget_heights = self.NN_lab_featureScale.size().height()
             
        self.NN_but_featureScaleVal = QPushButton(self.NN_scrollArea_widgetContents)
        self.NN_but_featureScaleVal.setObjectName("NN_but_featureScaleVal")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.FieldRole, self.NN_but_featureScaleVal)
        self.NN_but_featureScaleVal.setMaximumHeight(40)
        self.NN_but_featureScaleVal.setText("View")
        self.NN_but_featureScaleVal.clicked.connect(self.viewCurrentFeatureScale)
        
        NN_pos += 1     
        self.NN_lab_avgNormalise = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_avgNormalise.setObjectName("NN_lab_avgNormalise")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.LabelRole, self.NN_lab_avgNormalise)
        self.NN_lab_avgNormalise.setAlignment(Qt.AlignCenter)
        self.NN_lab_avgNormalise.setMaximumHeight(40)
        self.NN_lab_avgNormalise.setText("Average normalisation:")
        
        self.NN_but_avgNormalise = QPushButton(self.NN_scrollArea_widgetContents)
        self.NN_but_avgNormalise.setObjectName("NN_but_avgNormalise")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.FieldRole, self.NN_but_avgNormalise)
        self.NN_but_avgNormalise.setMaximumHeight(40)
        self.NN_but_avgNormalise.setText("View")
        self.NN_but_avgNormalise.clicked.connect(self.viewCurrentAvgNormalise)
        
        NN_pos += 1
        self.NN_line_arch = QFrame(self.NN_scrollArea_widgetContents)
        self.NN_line_arch.setFrameShape(QFrame.HLine)
        self.NN_line_arch.setFrameShadow(QFrame.Sunken)
        self.NN_line_arch.setObjectName("NN_line_arch")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_line_arch)
        
        NN_pos += 1     
        self.NN_title_arch = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_title_arch.setObjectName("NN_title_arch")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_title_arch)
        self.NN_title_arch.setAlignment(Qt.AlignCenter)
        self.NN_title_arch.setMaximumHeight(40)
        self.NN_title_arch.setStyleSheet("font: bold 11pt")
        self.NN_title_arch.setText("Architecture")
        
        NN_pos += 1     
        self.NN_lab_noLayers = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_noLayers.setObjectName("NN_lab_noLayers")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.LabelRole, self.NN_lab_noLayers)
        self.NN_lab_noLayers.setAlignment(Qt.AlignCenter)
        self.NN_lab_noLayers.setMaximumHeight(40)
        self.NN_lab_noLayers.setStyleSheet("font: 12pt")
        self.NN_lab_noLayers.setText("No. layers:")
        
        self.NN_lab_noLayersVal = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_noLayersVal.setObjectName("NN_lab_noLayersVal")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.FieldRole, self.NN_lab_noLayersVal)
        self.NN_lab_noLayersVal.setMaximumHeight(40)
        self.NN_lab_noLayersVal.setStyleSheet("font: 12pt")
        self.NN_lab_noLayersVal.setText("None")
        
        NN_pos += 1     
        self.NN_lab_arch = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_arch.setObjectName("NN_lab_arch")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.LabelRole, self.NN_lab_arch)
        self.NN_lab_arch.setAlignment(Qt.AlignCenter)
        self.NN_lab_arch.setMaximumHeight(40)
        self.NN_lab_arch.setStyleSheet("font: 12pt")
        self.NN_lab_arch.setText("Architecture:")
        
        self.NN_but_arch = QPushButton(self.NN_scrollArea_widgetContents)
        self.NN_but_arch.setObjectName("NN_but_arch")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.FieldRole, self.NN_but_arch)
        self.NN_but_arch.clicked.connect(self.viewCurrentArchitecture)
        self.NN_but_arch.setText("View")
               
        NN_pos += 1     
        self.NN_lab_arcFun = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_arcFun.setObjectName("NN_lab_arcFun")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.LabelRole, self.NN_lab_arcFun)
        self.NN_lab_arcFun.setAlignment(Qt.AlignCenter)
        self.NN_lab_arcFun.setMaximumHeight(40)
        self.NN_lab_arcFun.setStyleSheet("font: 12pt")
        self.NN_lab_arcFun.setText("Activation function:")
        
        self.NN_lab_arcFun_val = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_arcFun_val.setObjectName("NN_lab_arcFun_val")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.FieldRole, self.NN_lab_arcFun_val)
        self.NN_lab_arcFun_val.setMaximumHeight(40)
        self.NN_lab_arcFun_val.setStyleSheet("font: 12pt")
        self.NN_lab_arcFun_val.setText("None")
        
        NN_pos += 1
        self.NN_line_hyperParam = QFrame(self.NN_scrollArea_widgetContents)
        self.NN_line_hyperParam.setFrameShape(QFrame.HLine)
        self.NN_line_hyperParam.setFrameShadow(QFrame.Sunken)
        self.NN_line_hyperParam.setObjectName("NN_line_hyperParam")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_line_hyperParam)
        
        NN_pos += 1     
        self.NN_title_hyperParam = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_title_hyperParam.setObjectName("NN_title_hyperParam")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_title_hyperParam)
        self.NN_title_hyperParam.setAlignment(Qt.AlignCenter)
        self.NN_title_hyperParam.setMaximumHeight(40)
        self.NN_title_hyperParam.setStyleSheet("font: bold 11pt")
        self.NN_title_hyperParam.setText("Hyperparameters")
        
        NN_pos += 1     
        self.NN_lab_reg = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_reg.setObjectName("NN_lab_reg")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.LabelRole, self.NN_lab_reg)
        self.NN_lab_reg.setAlignment(Qt.AlignCenter)
        self.NN_lab_reg.setMaximumHeight(40)
        self.NN_lab_reg.setStyleSheet("font: 12pt")
        self.NN_lab_reg.setText("Regularisation parameter:")
        
        self.NN_lab_reg_val = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_reg_val.setObjectName("NN_lab_reg_val")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.FieldRole, self.NN_lab_reg_val)
        self.NN_lab_reg_val.setMaximumHeight(40)
        self.NN_lab_reg_val.setStyleSheet("font: 12pt")
        self.NN_lab_reg_val.setText("None")
        
        NN_pos += 1
        self.NN_line_pred = QFrame(self.NN_scrollArea_widgetContents)
        self.NN_line_pred.setFrameShape(QFrame.HLine)
        self.NN_line_pred.setFrameShadow(QFrame.Sunken)
        self.NN_line_pred.setObjectName("NN_line_pred")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_line_pred)
        
        NN_pos += 1     
        self.NN_title_pred = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_title_pred.setObjectName("NN_title_pred")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_title_pred)
        self.NN_title_pred.setAlignment(Qt.AlignCenter)
        self.NN_title_pred.setMaximumHeight(40)
        self.NN_title_pred.setStyleSheet("font: bold 11pt")
        self.NN_title_pred.setText("Predictions")
        
        NN_pos += 1
        self.NN_lab_predFileName = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_predFileName.setObjectName("NN_lab_predFileName")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.LabelRole, self.NN_lab_predFileName)
        self.NN_lab_predFileName.setText("Source file:")

        self.NN_but_predFileName = QPushButton(self.NN_scrollArea_widgetContents)
        self.NN_but_predFileName.setObjectName("NN_but_predFileName")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.FieldRole, self.NN_but_predFileName)
        self.NN_but_predFileName.clicked.connect(self.getPredFileName)
        self.NN_but_predFileName.setText("Browse")
        
        NN_pos += 1
        self.NN_lab_XPredSource = QLabel(self.NN_scrollArea_widgetContents)
        self.NN_lab_XPredSource.setObjectName("NN_lab_XPredSource")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.LabelRole, self.NN_lab_XPredSource)
        self.NN_lab_XPredSource.setText("X sheet:")

        self.NN_comboBox_XPredSource = QComboBox(self.NN_scrollArea_widgetContents)
        self.NN_comboBox_XPredSource.setObjectName("NN_comboBox_XPredSource")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.FieldRole, self.NN_comboBox_XPredSource)
        
        NN_pos += 1
        self.NN_checkBox_headers = QCheckBox(self.NN_scrollArea_widgetContents)
        self.NN_checkBox_headers.setObjectName("NN_checkBox_headers")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.NN_checkBox_headers)
        self.NN_checkBox_headers.setText("Column headers on data source sheet")
        
        NN_pos += 1
        self.but_predict = QPushButton(self.NN_scrollArea_widgetContents)
        self.but_predict.setObjectName("but_predict")
        self.NN_formLayout.setWidget(NN_pos, QFormLayout.SpanningRole, self.but_predict)
        self.but_predict.clicked.connect(self.predict)
        self.but_predict.setText("Predict")
        
        self.NN_scrollArea.setWidget(self.NN_scrollArea_widgetContents)
        self.horizontalLayoutSuper.addWidget(self.NN_scrollArea)
        ### Current NN
        
        ### newNN
        self.newNN_scrollArea = QScrollArea(self.main_scrollArea_widgetContents)
        self.newNN_scrollArea.setWidgetResizable(True)
        self.newNN_scrollArea.setObjectName("newNN_scrollArea")
        self.newNN_scrollArea.setFixedWidth(526)
        self.newNN_scrollArea_widgetContents = QWidget()
        self.newNN_scrollArea_widgetContents.setObjectName("newNN_scrollArea_widgetContents")
        self.newNN_scrollArea.setWidget(self.newNN_scrollArea_widgetContents)
        self.newNN_formLayout = QFormLayout(self.newNN_scrollArea_widgetContents)
        self.newNN_formLayout.setObjectName("newNN_formLayout")    
        self.newNN_formLayout.setAlignment(Qt.AlignTop)
        
        newNN_pos = 0
        self.newNN_title_top = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_title_top.setObjectName("newNN_title_top")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_title_top)
        self.newNN_title_top.setAlignment(Qt.AlignCenter)
        self.newNN_title_top.setMaximumHeight(40)
        self.newNN_title_top.setStyleSheet("font: bold 12pt")
        self.newNN_title_top.setText("Model New Neural Network")
        
        newNN_pos += 1
        self.newNN_line_top = QFrame(self.newNN_scrollArea_widgetContents)
        self.newNN_line_top.setFrameShape(QFrame.HLine)
        self.newNN_line_top.setFrameShadow(QFrame.Plain)
        self.newNN_line_top.setLineWidth(1)
        self.newNN_line_top.setObjectName("newNN_line_top")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_line_top)
        self.newNN_line_top.setPalette(self.palette)
        
        newNN_pos += 1
        self.newNN_title_dataPre = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_title_dataPre.setObjectName("newNN_title_dataPre")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_title_dataPre)
        self.newNN_title_dataPre.setAlignment(Qt.AlignCenter)
        self.newNN_title_dataPre.setMaximumHeight(40)
        self.newNN_title_dataPre.setStyleSheet("font: bold 11pt")
        self.newNN_title_dataPre.setText("Data Sourcing and Preprocessing")
        
        newNN_pos += 1
        self.newNN_lab_modelFileName = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_modelFileName.setObjectName("newNN_lab_modelFileName")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_modelFileName)
        self.newNN_lab_modelFileName.setText("Source file:")
        
        self.newNN_but_modelFileName = QPushButton(self.newNN_scrollArea_widgetContents)
        self.newNN_but_modelFileName.setObjectName("newNN_but_modelFileName")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_but_modelFileName)
        self.newNN_but_modelFileName.clicked.connect(self.getModelFileName)
        self.newNN_but_modelFileName.setText("Browse")
        
        newNN_pos += 1
        self.newNN_lab_XTrainSource = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_XTrainSource.setObjectName("newNN_lab_XTrainSource")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_XTrainSource)
        self.newNN_lab_XTrainSource.setText("X sheet:")

        self.newNN_comboBox_XTrainSource = QComboBox(self.newNN_scrollArea_widgetContents)
        self.newNN_comboBox_XTrainSource.setObjectName("newNN_comboBox_XTrainSource")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_comboBox_XTrainSource)
        
        newNN_pos += 1
        self.newNN_lab_YTrainSource = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_YTrainSource.setObjectName("newNN_lab_YTrainSource")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_YTrainSource)
        self.newNN_lab_YTrainSource.setText("Y sheet:")

        self.newNN_comboBox_YTrainSource = QComboBox(self.newNN_scrollArea_widgetContents)
        self.newNN_comboBox_YTrainSource.setObjectName("newNN_comboBox_YTrainSource")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_comboBox_YTrainSource)
        
        newNN_pos += 1
        self.newNN_checkBox_headers = QCheckBox(self.newNN_scrollArea_widgetContents)
        self.newNN_checkBox_headers.setObjectName("newNN_checkBox_headers")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_checkBox_headers)
        self.newNN_checkBox_headers.setText("Column headers on data source sheets")
        
        newNN_pos += 1
        self.newNN_lab_featureScale = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_featureScale.setObjectName("newNN_lab_featureScale")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_featureScale)
        self.newNN_lab_featureScale.setText("Feature scaling:")
        
        self.newNN_comboBox_featureScale = QComboBox(self.newNN_scrollArea_widgetContents)
        self.newNN_comboBox_featureScale.setObjectName("newNN_comboBox_featureScale")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_comboBox_featureScale)
        self.newNN_comboBox_featureScale.addItem("")
        self.newNN_comboBox_featureScale.addItem("")
        self.newNN_comboBox_featureScale.addItem("")
        self.newNN_comboBox_featureScale.addItem("")
        self.newNN_comboBox_featureScale.setItemText(0, "Variance")
        self.newNN_comboBox_featureScale.setItemText(1, "Standard Deviation")
        self.newNN_comboBox_featureScale.setItemText(2, "Range")
        self.newNN_comboBox_featureScale.setItemText(3, "None")
        
        newNN_pos += 1
        self.newNN_lab_avgNormalise = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_avgNormalise.setObjectName("newNN_lab_avgNormalise")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_avgNormalise)
        self.newNN_lab_avgNormalise.setText("Average normalisation:")
        
        self.newNN_comboBox_avgNormalise = QComboBox(self.newNN_scrollArea_widgetContents)
        self.newNN_comboBox_avgNormalise.setObjectName("newNN_comboBox_avgNormalise")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_comboBox_avgNormalise)
        self.newNN_comboBox_avgNormalise.addItem("")
        self.newNN_comboBox_avgNormalise.addItem("")
        self.newNN_comboBox_avgNormalise.addItem("")
        self.newNN_comboBox_avgNormalise.addItem("")
        self.newNN_comboBox_avgNormalise.setItemText(0, "Mean")
        self.newNN_comboBox_avgNormalise.setItemText(1, "Median")
        self.newNN_comboBox_avgNormalise.setItemText(2, "Mode")
        self.newNN_comboBox_avgNormalise.setItemText(3, "None")
        
        newNN_pos += 1
        self.newNN_line_dataSplit = QFrame(self.newNN_scrollArea_widgetContents)
        self.newNN_line_dataSplit.setFrameShape(QFrame.HLine)
        self.newNN_line_dataSplit.setFrameShadow(QFrame.Sunken)
        self.newNN_line_dataSplit.setObjectName("newNN_line_dataSplit")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_line_dataSplit)
        
        newNN_pos += 1
        self.newNN_lab_train = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_train.setObjectName("newNN_lab_train")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_train)
        self.newNN_lab_train.setText("Training data split:")
        
        self.newNN_spinBox_train = QDoubleSpinBox(self.newNN_scrollArea_widgetContents)
        self.newNN_spinBox_train.setRange(0,1)
        self.newNN_spinBox_train.setSingleStep(0.1)
        self.newNN_spinBox_train.setDecimals(3)
        self.newNN_spinBox_train.setValue(0.6)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_spinBox_train)
        
        newNN_pos += 1
        self.newNN_lab_CV = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_CV.setObjectName("newNN_lab_CV")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_CV)
        self.newNN_lab_CV.setText("Cross validation data split:")
        
        self.newNN_spinBox_CV = QDoubleSpinBox(self.newNN_scrollArea_widgetContents)
        self.newNN_spinBox_CV.setObjectName("newNN_spinBox_CV")
        self.newNN_spinBox_CV.setRange(0,1)
        self.newNN_spinBox_CV.setSingleStep(0.1)
        self.newNN_spinBox_CV.setDecimals(3)
        self.newNN_spinBox_CV.setValue(0.2)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_spinBox_CV)
        
        newNN_pos += 1
        self.newNN_lab_test = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_test.setObjectName("newNN_lab_test")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_test)
        self.newNN_lab_test.setText("Test data split:")
        
        self.newNN_spinBox_test = QDoubleSpinBox(self.newNN_scrollArea_widgetContents)
        self.newNN_spinBox_test.setObjectName("newNN_spinBox_test")
        self.newNN_spinBox_test.setRange(0,1)
        self.newNN_spinBox_test.setSingleStep(0.1)
        self.newNN_spinBox_test.setDecimals(3)
        self.newNN_spinBox_test.setValue(0.2)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_spinBox_test)
        
        newNN_pos += 1
        self.newNN_line_NNParams = QFrame(self.newNN_scrollArea_widgetContents)
        self.newNN_line_NNParams.setFrameShape(QFrame.HLine)
        self.newNN_line_NNParams.setFrameShadow(QFrame.Sunken)
        self.newNN_line_NNParams.setObjectName("newNN_line_NNParams")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_line_NNParams)
        
        newNN_pos += 1
        self.newNN_title_NNParams = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_title_NNParams.setObjectName("newNN_title_NNParams")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_title_NNParams)
        self.newNN_title_NNParams.setAlignment(Qt.AlignCenter)
        self.newNN_title_NNParams.setMaximumHeight(40)
        self.newNN_title_NNParams.setStyleSheet("font: bold 11pt")
        self.newNN_title_NNParams.setText("Neural Network Parameters")
        
        newNN_pos += 1
        self.newNN_lab_noLayers = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_noLayers.setObjectName("newNN_lab_noLayers")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_noLayers)
        self.newNN_lab_noLayers.setText("No. layers:")
                
        self.newNN_spinBox_noLayers = QSpinBox(self.newNN_scrollArea_widgetContents)
        self.newNN_spinBox_noLayers.setObjectName("newNN_spinBox_noLayers")
        self.newNN_spinBox_noLayers.setMinimum(3)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_spinBox_noLayers)
        self.newNN_spinBox_noLayers.valueChanged.connect(self.changeLayers)
        self.newNN_spinBox_noLayersprevVal = 0
        
        newNN_pos += 1
        self.newNN_lab_weightInit = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_weightInit.setObjectName("newNN_lab_weightInit")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_weightInit)
        self.newNN_lab_weightInit.setText("Weight initialisation:")
        
        self.newNN_comboBox_weightInit = QComboBox(self.newNN_scrollArea_widgetContents)
        self.newNN_comboBox_weightInit.setObjectName("newNN_comboBox_weightInit")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_comboBox_weightInit)
        self.newNN_comboBox_weightInit.addItem("")
        self.newNN_comboBox_weightInit.addItem("")
        self.newNN_comboBox_weightInit.addItem("")
        self.newNN_comboBox_weightInit.setItemText(0, "Classic")
        self.newNN_comboBox_weightInit.setItemText(1, "Xavier")
        self.newNN_comboBox_weightInit.setItemText(2, "Kaiming")
        self.newNN_comboBox_weightInit.setCurrentIndex(1)
        
        newNN_pos += 1
        self.newNN_lab_actFunKind = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_actFunKind.setObjectName("newNN_lab_actFunKind")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_actFunKind)
        self.newNN_lab_actFunKind.setText("Activation function:")
        
        self.newNN_comboBox_actFunKind = QComboBox(self.newNN_scrollArea_widgetContents)
        self.newNN_comboBox_actFunKind.setObjectName("newNN_comboBox_actFunKind")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_comboBox_actFunKind)
        self.newNN_comboBox_actFunKind.addItem("")
        # self.newNN_comboBox_actFunKind.addItem("")
        # self.newNN_comboBox_actFunKind.addItem("")
        self.newNN_comboBox_actFunKind.setItemText(0, "Sigmoid")
        # self.newNN_comboBox_actFunKind.setItemText(1, "ReLU")
        # self.newNN_comboBox_actFunKind.setItemText(2, "Heaviside")
        
        newNN_pos += 1
        self.newNN_lab_regVals = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_regVals.setObjectName("newNN_lab_regVals")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_regVals)
        self.newNN_lab_regVals.setText("Regularisation parameter(s):")
        
        self.newNN_tex_regVals = QPlainTextEdit(self.newNN_scrollArea_widgetContents)
        self.newNN_tex_regVals.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.newNN_tex_regVals.setObjectName("newNN_tex_regVals")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_tex_regVals)
        self.newNN_tex_regVals.setMaximumHeight(self.widget_heights+30)
        self.newNN_tex_regVals.setPlaceholderText("0.01,0.03,0.1,...")
        self.newNN_tex_regVals.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
        
        newNN_pos += 1
        self.newNN_line_optParam = QFrame(self.newNN_scrollArea_widgetContents)
        self.newNN_line_optParam.setFrameShape(QFrame.HLine)
        self.newNN_line_optParam.setFrameShadow(QFrame.Sunken)
        self.newNN_line_optParam.setObjectName("newNN_line_optParam")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_line_optParam)
        
        newNN_pos += 1
        self.newNN_title_optParam = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_title_optParam.setObjectName("newNN_title_optParam")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_title_optParam)
        self.newNN_title_optParam.setAlignment(Qt.AlignCenter)
        self.newNN_title_optParam.setMaximumHeight(40)
        self.newNN_title_optParam.setStyleSheet("font: bold 11pt")
        self.newNN_title_optParam.setText("Optimiser Parameters")
        
        newNN_pos += 1
        self.newNN_lab_noInitLoops = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_noInitLoops.setObjectName("newNN_lab_noInitLoops")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_noInitLoops)
        self.newNN_lab_noInitLoops.setText("No. weight initialisations:")
        
        self.newNN_spinBox_noInitLoops = QSpinBox(self.newNN_scrollArea_widgetContents)
        self.newNN_spinBox_noInitLoops.setObjectName("newNN_spinBox_noInitLoops")
        self.newNN_spinBox_noInitLoops.setMinimum(1)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_spinBox_noInitLoops)
        
        newNN_pos += 1
        self.newNN_lab_noShuffleLoops = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_noShuffleLoops.setObjectName("newNN_lab_noShuffleLoops")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_noShuffleLoops)
        self.newNN_lab_noShuffleLoops.setText("No. training data shuffles:")
        
        self.newNN_spinBox_noShuffleLoops = QSpinBox(self.newNN_scrollArea_widgetContents)
        self.newNN_spinBox_noShuffleLoops.setObjectName("newNN_spinBox_noShuffleLoops")
        self.newNN_spinBox_noShuffleLoops.setMinimum(1)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_spinBox_noShuffleLoops)
        
        newNN_pos += 1
        self.newNN_lab_optimiserKind = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_optimiserKind.setObjectName("newNN_lab_optimiserKind")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_optimiserKind)
        self.newNN_lab_optimiserKind.setText("Optimiser algorithm:")
        
        self.newNN_comboBox_optimiserKind = QComboBox(self.newNN_scrollArea_widgetContents)
        self.newNN_comboBox_optimiserKind.setObjectName("newNN_comboBox_optimiserKind")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_comboBox_optimiserKind)
        self.newNN_comboBox_optimiserKind.addItem("")
        self.newNN_comboBox_optimiserKind.addItem("")
        self.newNN_comboBox_optimiserKind.addItem("")
        self.newNN_comboBox_optimiserKind.setItemText(0, "Gradient descent")
        self.newNN_comboBox_optimiserKind.setItemText(1, "Steepest descents")
        self.newNN_comboBox_optimiserKind.setItemText(2, "Conjugate gradients")
        self.newNN_comboBox_optimiserKind.setCurrentIndex(0)
        
        newNN_pos += 1
        self.newNN_lab_maxIters = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_maxIters.setObjectName("newNN_lab_maxIters")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_maxIters)
        self.newNN_lab_maxIters.setText("Max iterations:")
        
        self.newNN_spinBox_maxIters = QSpinBox(self.newNN_scrollArea_widgetContents)
        self.newNN_spinBox_maxIters.setObjectName("newNN_spinBox_maxIters")
        self.newNN_spinBox_maxIters.setRange(1,10**9)
        self.newNN_spinBox_maxIters.setSingleStep(1000)
        self.newNN_spinBox_maxIters.setValue(1000)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_spinBox_maxIters)
        
        newNN_pos += 1
        self.newNN_lab_tol = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_tol.setObjectName("newNN_lab_tol")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_tol)
        self.newNN_lab_tol.setText("Tolerance:")
        
        self.newNN_spinBox_tol = QDoubleSpinBox(self.newNN_scrollArea_widgetContents)
        self.newNN_spinBox_tol.setObjectName("newNN_spinBox_tol")
        self.newNN_spinBox_tol.setMinimum(10**-9)
        self.newNN_spinBox_tol.setSingleStep(10**-6)
        self.newNN_spinBox_tol.setDecimals(9)
        self.newNN_spinBox_tol.setValue(10**-6)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_spinBox_tol)
        
        newNN_pos += 1
        gradDesc_pos = 0
        self.newNN_gradDesc = []
        self.newNN_gradDesc.append(QLabel(self.newNN_scrollArea_widgetContents))
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_gradDesc[gradDesc_pos])
        self.newNN_gradDesc[gradDesc_pos].setText("Step size:")
        
        gradDesc_pos += 1
        self.newNN_gradDesc.append(QDoubleSpinBox(self.newNN_scrollArea_widgetContents))
        self.newNN_gradDesc[gradDesc_pos].setMinimum(0.001)
        self.newNN_gradDesc[gradDesc_pos].setSingleStep(0.1)
        self.newNN_gradDesc[gradDesc_pos].setDecimals(3)
        self.newNN_gradDesc[gradDesc_pos].setValue(1)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_gradDesc[gradDesc_pos])
        
        newNN_pos += 1
        gradDesc_pos += 1
        self.newNN_gradDesc.append(QLabel(self.newNN_scrollArea_widgetContents))
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_gradDesc[gradDesc_pos])
        self.newNN_gradDesc[gradDesc_pos].setText("Step size change factor:")
        
        gradDesc_pos += 1
        self.newNN_gradDesc.append(QDoubleSpinBox(self.newNN_scrollArea_widgetContents))
        self.newNN_gradDesc[gradDesc_pos].setMinimum(0)
        self.newNN_gradDesc[gradDesc_pos].setSingleStep(0.1)
        self.newNN_gradDesc[gradDesc_pos].setDecimals(2)
        self.newNN_gradDesc[gradDesc_pos].setValue(0)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_gradDesc[gradDesc_pos])
        
        newNN_pos += 1
        steepConj_pos = 0
        self.newNN_steepConj = []
        self.newNN_steepConj.append(QLabel(self.newNN_scrollArea_widgetContents))
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_steepConj[steepConj_pos])
        self.newNN_steepConj[steepConj_pos].setText("Step optimiser max iters:")
        
        steepConj_pos += 1
        self.newNN_steepConj.append(QSpinBox(self.newNN_scrollArea_widgetContents))
        self.newNN_steepConj[steepConj_pos].setRange(1,10**9)
        self.newNN_steepConj[steepConj_pos].setSingleStep(10)
        self.newNN_steepConj[steepConj_pos].setValue(10)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_steepConj[steepConj_pos])
        
        newNN_pos += 1
        steepConj_pos += 1
        self.newNN_steepConj.append(QLabel(self.newNN_scrollArea_widgetContents))
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_steepConj[steepConj_pos])
        self.newNN_steepConj[steepConj_pos].setText("Step optimiser tolerance:")
        
        steepConj_pos += 1
        self.newNN_steepConj.append(QDoubleSpinBox(self.newNN_scrollArea_widgetContents))
        self.newNN_steepConj[steepConj_pos].setMinimum(10**-9)
        self.newNN_steepConj[steepConj_pos].setSingleStep(10**-9)
        self.newNN_steepConj[steepConj_pos].setDecimals(9)
        self.newNN_steepConj[steepConj_pos].setValue(10**-9)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_steepConj[steepConj_pos])

        self.newNN_comboBox_optimiserKind.currentIndexChanged.connect(self.optimiserChanged)
        self.newNN_comboBox_optimiserKind.setCurrentIndex(2)
        self.newNN_comboBox_optimiserKind.setCurrentIndex(0)
        
        newNN_pos += 1
        self.newNN_line_checkMeth = QFrame(self.newNN_scrollArea_widgetContents)
        self.newNN_line_checkMeth.setFrameShape(QFrame.HLine)
        self.newNN_line_checkMeth.setFrameShadow(QFrame.Sunken)
        self.newNN_line_checkMeth.setObjectName("newNN_line_checkMeth")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_line_checkMeth)
        
        newNN_pos += 1
        self.newNN_title_checkMeth = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_title_checkMeth.setObjectName("newNN_title_checkMeth")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_title_checkMeth)
        self.newNN_title_checkMeth.setAlignment(Qt.AlignCenter)
        self.newNN_title_checkMeth.setMaximumHeight(40)
        self.newNN_title_checkMeth.setStyleSheet("font: bold 11pt")
        self.newNN_title_checkMeth.setText("Checking Methods")
        
        newNN_pos += 1
        self.newNN_checkBox_gradCheck = QCheckBox(self.newNN_scrollArea_widgetContents)
        self.newNN_checkBox_gradCheck.setObjectName("newNN_checkBox_gradCheck")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_checkBox_gradCheck)
        self.newNN_checkBox_gradCheck.setText("Gradient check")
        self.newNN_checkBox_gradCheck.stateChanged.connect(self.gradCheckChanged)
        
        self.newNN_checkBox_costsPlot = QCheckBox(self.newNN_scrollArea_widgetContents)
        self.newNN_checkBox_costsPlot.setObjectName("newNN_checkBox_costsPlot")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_checkBox_costsPlot)
        self.newNN_checkBox_costsPlot.setText("Display costs plot")
        
        newNN_pos+=1
        self.newNN_lab_gradCheckEps = QLabel(self.newNN_scrollArea_widgetContents)
        self.newNN_lab_gradCheckEps.setObjectName("newNN_lab_gradCheckEps")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.LabelRole, self.newNN_lab_gradCheckEps)
        self.newNN_lab_gradCheckEps.setText("Gradient check resolution:")
        
        self.newNN_spinBox_gradCheckEps = QDoubleSpinBox(self.newNN_scrollArea_widgetContents)
        self.newNN_spinBox_gradCheckEps.setObjectName("newNN_spinBox_gradCheckEps")
        self.newNN_spinBox_gradCheckEps.setMinimum(10**-9)
        self.newNN_spinBox_gradCheckEps.setSingleStep(10**-6)
        self.newNN_spinBox_gradCheckEps.setDecimals(9)
        self.newNN_spinBox_gradCheckEps.setValue(10**-6)
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.FieldRole, self.newNN_spinBox_gradCheckEps)
        self.newNN_checkBox_gradCheck.setChecked(True)
        self.newNN_checkBox_gradCheck.setChecked(False)
        
        newNN_pos += 1
        self.newNN_line_model = QFrame(self.newNN_scrollArea_widgetContents)
        self.newNN_line_model.setFrameShape(QFrame.HLine)
        self.newNN_line_model.setFrameShadow(QFrame.Sunken)
        self.newNN_line_model.setObjectName("newNN_line_model")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.newNN_line_model)
        
        newNN_pos += 1
        self.but_model = QPushButton(self.newNN_scrollArea_widgetContents)
        self.but_model.setObjectName("but_model")
        self.newNN_formLayout.setWidget(newNN_pos, QFormLayout.SpanningRole, self.but_model)
        self.but_model.clicked.connect(self.model)
        self.but_model.setText("Model")           
        
        self.newNN_scrollArea.setWidget(self.newNN_scrollArea_widgetContents)
        self.horizontalLayoutSuper.addWidget(self.newNN_scrollArea)
        ### newNN
        
        self.horizontalLayoutSuper.addWidget(self.main_scrollArea)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.changeLayers()
    
    ### Catches errors and displays in message box (not all errors are informatively displayed)
    def errorCatcher(f):
        def inner(self):
            try:
                return f(self)
            except Exception as e:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                print(str(e))
                msg.setText(str(e))
                msg.setIcon(QMessageBox.Warning)
                msg.setStandardButtons(QMessageBox.Ok)
                self.loading = False
                msg.exec_()
                return
        return inner    
    
    ### Changes optimiser display if different ones selected
    def optimiserChanged(self):
        if self.newNN_comboBox_optimiserKind.currentText() == 'Gradient descent':
            for obj in self.newNN_gradDesc:
                obj.setVisible(True)
            for obj in self.newNN_steepConj:
                obj.setVisible(False)
            self.newNN_spinBox_maxIters.setValue(1000)    
        elif self.newNN_comboBox_optimiserKind.currentText() == 'Steepest descents' or self.newNN_comboBox_optimiserKind.currentText() == 'Conjugate gradients':
            for obj in self.newNN_gradDesc:
                obj.setVisible(False)
            for obj in self.newNN_steepConj:
                obj.setVisible(True)
            self.newNN_spinBox_maxIters.setValue(100)
            
    ### Changes display if want to do gradient check
    def gradCheckChanged(self):
        if self.newNN_checkBox_gradCheck.isChecked():
            self.newNN_lab_gradCheckEps.setVisible(True)
            self.newNN_spinBox_gradCheckEps.setVisible(True)
        else:
            self.newNN_lab_gradCheckEps.setVisible(False)
            self.newNN_spinBox_gradCheckEps.setVisible(False)
    
    ### Upload training data
    @errorCatcher
    def getModelFileName(self):
        self.modelFileFullName, _ = QFileDialog.getOpenFileName(self, "Open Prediction Source File", os.getcwd(), "(*.xlsx)")
        self.modelFileName = re.split(':|/',self.modelFileFullName)[-1]
        if self.modelFileFullName == '':
            return
        self.startLoadingAnimation("Getting file")
        self.newNN_but_modelFileName.setText(self.modelFileName)
        tempExcel = pd.ExcelFile(self.modelFileFullName).sheet_names
        for i in range(0,len(tempExcel)):
            self.newNN_comboBox_XTrainSource.addItem("")
            self.newNN_comboBox_XTrainSource.setItemText(i, tempExcel[i])
            self.newNN_comboBox_YTrainSource.addItem("")
            self.newNN_comboBox_YTrainSource.setItemText(i, tempExcel[i])
        self.loading = False
    
    ### Uploading prediction data
    @errorCatcher            
    def getPredFileName(self):
        self.predFileFullName, _ = QFileDialog.getOpenFileName(self, "Open Prediction Source File", os.getcwd(), "(*.xlsx)")
        self.predFileName = re.split(':|/',self.predFileFullName)[-1]
        if self.predFileFullName == '':
            return
        self.startLoadingAnimation("Getting file")
        self.NN_but_predFileName.setText(self.predFileName)
        tempExcel = pd.ExcelFile(self.predFileFullName).sheet_names
        for i in range(0,len(tempExcel)):
            self.NN_comboBox_XPredSource.addItem("")
            self.NN_comboBox_XPredSource.setItemText(i, tempExcel[i])
        self.loading = False

    ### View instruction window - detailed enough?
    def viewInstructions(self):
        self.instructionsWindow = textWindow('Instructions',
                                             '1. Create an excel file with your numerical training data:\n'
                                             '    1a. One sheet must have your input, \'X\', data and the other the output, \'Y\', data.\n'
                                             '    1b. Each sheet must have each record as a row, and each attribute as a column.\n'
                                             '    1c. The output sheet\'s columns denote each category and the values must be either a \'1\' or a \'0\' to denote whether the record is a member of that category.\n'
                                             '    1d. There may be a single row for data headers, in which case the relevant checkbox must be selected.\n'
                                             '2. Upload the excel file under the \'Model New Neural Network\' heading and choose the relevant sheet names.\n'
                                             '3. Play around with the fields under the \'Model New Neural Network\' heading and in the neural network designer.\n'
                                             '4. Click \'Model\' to create and train your neural network.\n'
                                             '    4a. If this seems like it is taking a while, check the black console box to make sure the program is infact still computing.\n'
                                             '5. Once modelled, it will automatically choose the model with the best hyperparameters of those given and display them under the \'Current Neural Network\' heading.\n'
                                             '6. You can now use this to predict the outcome of new events!\n'
                                             '7. If you like your neural network, you can save it for later and use it again in future!')

    ### View window displaying average normalisation values
    def viewCurrentAvgNormalise(self):
        if 'my_NN' in dir(self):
            self.avgNormaliseWindow = textWindow('Average normalisation',
                                                 'Average normalisation of input columns:\n'+str(self.my_NN.avgNormalisation))
    
    ### View window displaying feature scaling values
    def viewCurrentFeatureScale(self):
        if 'my_NN' in dir(self):
            self.featureScaleWindow = textWindow('Feature scaling',
                                                 'Feature scaling of input columns:\n'+str(self.my_NN.featureScaling))
    
    ### View window displaying current architecture
    def viewCurrentArchitecture(self):
        if 'my_NN' in dir(self):
            self.architectureWindow = architectureWindow(self.my_NN)
    
    ### Browse files to save NN
    def saveNN(self,As=False):
        if 'my_NN' not in dir(self):
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Neural Network not yet modelled")
            msg.setWindowIcon(QIcon(self.iconName))
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        if As or self.NN_savedName is None:
            self.NN_savedFullName, _ = QFileDialog.getSaveFileName(self,"Save Neural Network",os.getcwd(),"(*.NN)")
            self.NN_savedName = re.split(':|/',self.NN_savedFullName)[-1]
            if self.NN_savedFullName == '':
                return
            with open(self.NN_savedFullName,'wb') as NN_file:
                pickle.dump(self.my_NN, NN_file)
            self.setWindowTitle("Neural Network - "+ str(self.NN_savedName))
        else:
            with open(self.NN_savedFullName,'wb') as NN_file:
                pickle.dump(self.my_NN, NN_file)
    
    ### Browse files to upload NN
    @errorCatcher
    def uploadNN(self):
        self.NN_savedFullName, _ = QFileDialog.getOpenFileName(self, "Open Neural Network", os.getcwd(), "(*.NN)")
        if self.NN_savedFullName == '':
            return
        self.startLoadingAnimation("Uploading")
        self.NN_savedName = re.split(':|/',self.NN_savedFullName)[-1]
        with open(self.NN_savedFullName, 'rb') as NN_file:
            self.my_NN = pickle.load(NN_file)
        self.newNN_comboBox_actFunKind.setCurrentText(self.my_NN.activationKind)
        self.newNN_spinBox_noLayers.setValue(self.my_NN.noLayers)
        for i in range(0,self.my_NN.noLayers):
            self.layer_checkBox_bias[i].setChecked(self.my_NN.biases[i])
            self.layer_spinBox_nodes[i].setValue(self.my_NN.layers[i].no_nodes)
        self.updateNN()
        self.setWindowTitle("Neural Network - "+ str(self.NN_savedName))
        self.loading = False
    
    ### Change number of layers in display
    def changeLayers(self):
        if self.newNN_spinBox_noLayersprevVal == 0:
            self.layer_scrollArea = []
            self.layer_scrollArea_widgetContents = []
            self.layer_verticalLayouts = []
            self.layer_labelIndex = []
            self.layer_line_title = []
            self.layer_checkBox_bias = []
            self.layer_spinBox_nodes = []
            self.layer_line_2 = []
            self.layer_spinBox_nodes_prevVal = [0,0,0]
            self.layer_node_bias = []
            self.layer_nodes = []
            for i in range(0,3):
                self.initialiseLayer(i)
            self.newNN_spinBox_noLayersprevVal = 3
        else:
            for i in range(self.newNN_spinBox_noLayersprevVal,self.newNN_spinBox_noLayers.value()):
                self.initialiseLayer(i)
            for i in range(self.newNN_spinBox_noLayers.value(),self.newNN_spinBox_noLayersprevVal):
                self.removeLayer()   
            self.newNN_spinBox_noLayersprevVal = self.newNN_spinBox_noLayers.value()
    
    ### Change number of nodes in a layer
    def changeNodes(self,index):
        if self.biasChanged:
            self.layer_spinBox_nodes_prevVal[index] = self.layer_spinBox_nodes[index].value()
            return
        if self.layer_spinBox_nodes_prevVal[index] == 0:
            for j in range(0,self.layer_spinBox_nodes[index].value()-1 if self.layer_checkBox_bias[index] else self.layer_spinBox_nodes[index].value()):
                self.addNode(index,j)
                del self.layer_spinBox_nodes_prevVal[-1]
        else:
            prevVal = self.layer_spinBox_nodes_prevVal[index]-1 if self.layer_checkBox_bias[index].isChecked() else self.layer_spinBox_nodes_prevVal[index]
            nowVal = self.layer_spinBox_nodes[index].value()-1 if self.layer_checkBox_bias[index].isChecked() else self.layer_spinBox_nodes[index].value()
            for j in range(prevVal,nowVal):
                self.addNode(index,j)
            for j in range(nowVal,prevVal):
                self.layer_nodes[index][-1].setParent(None)
                del self.layer_nodes[index][-1]
        self.layer_spinBox_nodes_prevVal[index] = self.layer_spinBox_nodes[index].value()
     
    ### Add a node to a layer
    def addNode(self,i,j):
        self.layer_nodes[i].append(QLabel(self.layer_scrollArea_widgetContents[i]))
        self.layer_nodes[i][j].setText(str(j+1))
        self.layer_nodes[i][j].setStyleSheet("border: 8px solid #8888FF; border-radius: 50px;")
        self.layer_nodes[i][j].setFixedSize(100,100)
        self.layer_nodes[i][j].setAlignment(Qt.AlignCenter)
        self.layer_verticalLayouts[i].addWidget(self.layer_nodes[i][j])
    
    ### Change whether bias node visible in a layer
    def bias_visible(self,index):
        self.biasChanged = True
        if self.layer_checkBox_bias[index].isChecked():
            self.layer_node_bias[index].show()
            self.layer_spinBox_nodes[index].setValue(self.layer_spinBox_nodes[index].value()+1)
            self.layer_spinBox_nodes[index].setMinimum(2)
        else:
            self.layer_node_bias[index].hide()
            self.layer_spinBox_nodes[index].setMinimum(1)
            self.layer_spinBox_nodes[index].setValue(self.layer_spinBox_nodes[index].value()-1)
        self.biasChanged = False
    
    ### Initialises a new layer
    def initialiseLayer(self,index):
        self.layer_scrollArea.append(QScrollArea(self.main_scrollArea_widgetContents)) 
        self.layer_scrollArea[index].setWidgetResizable(True)
        self.layer_scrollArea[index].setMinimumWidth(178)
        self.layer_scrollArea_widgetContents.append(QWidget())
        self.layer_verticalLayouts.append(QtWidgets.QVBoxLayout(self.layer_scrollArea_widgetContents[index]))
        self.layer_verticalLayouts[index].setObjectName("verticalLayout")
        self.layer_verticalLayouts[index].setAlignment(Qt.AlignTop|Qt.AlignHCenter)
        
        self.layer_labelIndex.append(QLabel(self.layer_scrollArea_widgetContents[index]))
        self.layer_labelIndex[index].setText(str(index+1))
        self.layer_labelIndex[index].setStyleSheet("font: bold 11pt;")
        self.layer_labelIndex[index].setAlignment(Qt.AlignCenter)
        self.layer_labelIndex[index].setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred)
        self.layer_labelIndex[index].setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
        self.layer_verticalLayouts[index].addWidget(self.layer_labelIndex[index])
        
        self.layer_line_title.append(QFrame(self.layer_scrollArea_widgetContents[index]))
        self.layer_line_title[index].setFrameShape(QFrame.HLine)
        self.layer_line_title[index].setFrameShadow(QFrame.Plain)
        self.layer_line_title[index].setLineWidth(1)
        self.layer_line_title[index].setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
        self.layer_verticalLayouts[index].addWidget(self.layer_line_title[index])
        self.layer_line_title[index].setPalette(self.palette)
        
        self.layer_checkBox_bias.append(QCheckBox(self.layer_scrollArea_widgetContents[index]))
        self.layer_checkBox_bias[index].setText("Bias node")
        self.layer_checkBox_bias[index].setChecked(True)
        self.layer_checkBox_bias[index].stateChanged.connect(lambda:self.bias_visible(index))
        self.layer_checkBox_bias[index].setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
        self.layer_verticalLayouts[index].addWidget(self.layer_checkBox_bias[index])

        self.layer_spinBox_nodes.append(QSpinBox(self.layer_scrollArea_widgetContents[index]))
        self.layer_spinBox_nodes[index].setMinimum(1)
        self.layer_spinBox_nodes[index].setMaximum(2501)
        self.layer_spinBox_nodes[index].setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred)
        self.layer_spinBox_nodes[index].setPrefix('No. nodes: ')
        self.layer_verticalLayouts[index].addWidget(self.layer_spinBox_nodes[index])
        self.layer_spinBox_nodes_prevVal.append(self.layer_spinBox_nodes[index].value())
        
        self.layer_line_2.append(QFrame(self.layer_scrollArea_widgetContents[index]))
        self.layer_line_2[index].setFrameShape(QFrame.HLine)
        self.layer_line_2[index].setFrameShadow(QFrame.Sunken)
        self.layer_line_2[index].setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
        self.layer_verticalLayouts[index].addWidget(self.layer_line_2[index])
        
        self.layer_node_bias.append(QLabel(self.layer_scrollArea_widgetContents[index]))
        self.layer_node_bias[index].setText("0")
        self.layer_node_bias[index].setStyleSheet("border: 8px solid #FF88FF; border-radius: 50px;")
        self.layer_node_bias[index].setFixedSize(100,100)
        self.layer_node_bias[index].setAlignment(Qt.AlignCenter)
        self.layer_node_bias[index].setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
        node_bias_retainSize = self.layer_node_bias[index].sizePolicy()
        node_bias_retainSize.setRetainSizeWhenHidden(True)
        self.layer_node_bias[index].setSizePolicy(node_bias_retainSize)
        self.bias_visible(index)
        self.layer_verticalLayouts[index].addWidget(self.layer_node_bias[index])
        
        self.layer_nodes.append([])
        self.changeNodes(index)
        self.layer_spinBox_nodes[index].valueChanged.connect(lambda:self.changeNodes(index))
                
        self.layer_scrollArea[index].setWidget(self.layer_scrollArea_widgetContents[index])
        self.horizontalLayout.addWidget(self.layer_scrollArea[index])
    
    ### Removes a layer from display
    def removeLayer(self):
        self.layer_scrollArea[-1].setParent(None)
        self.layer_scrollArea_widgetContents[-1].setParent(None)
        del self.layer_scrollArea[-1]
        del self.layer_scrollArea_widgetContents[-1]
        del self.layer_verticalLayouts[-1]
        del self.layer_labelIndex[-1]
        del self.layer_line_title[-1]
        del self.layer_checkBox_bias[-1]
        del self.layer_spinBox_nodes[-1]
        del self.layer_line_2[-1]
        del self.layer_spinBox_nodes_prevVal[-1]
        del self.layer_node_bias[-1]
        del self.layer_nodes[-1]
    
    ### Start the loading animation
    def startLoadingAnimation(self,text):
        self.loading = True
        self.loadingThread = threading.Thread(target=self.loadingAnimation,args=(text,))
        self.loadingThread.daemon = True
        self.loadingThread.start()
        
    ### Loading animation
    def loadingAnimation(self,text):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if not self.loading:
                self.loading = True
                break
            sys.stdout.write('\r'+text+' ' + c)
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rFinished '+text.lower()+'!\n')
    
    ### Models NN
    @errorCatcher
    def model(self):
        self.startLoadingAnimation("Modelling")
        
        start_time = time.time()
        modelHeadersOn = 0 if self.newNN_checkBox_headers.isChecked() else None
        disp_gradCheck = self.newNN_checkBox_gradCheck.isChecked()
        disp_costs = self.newNN_checkBox_costsPlot.isChecked()
        if 'modelFileFullName' not in dir(self) or self.modelFileFullName == '':
            raise ValueError('No file found.')
        
        X_train,\
            Y_train,\
                _,\
                n_train,\
                    K_train,\
                        avgs_train,\
                            scales_train = NN.extract_data(pd.read_excel(self.modelFileFullName,
                                                                         sheet_name=self.newNN_comboBox_XTrainSource.currentText(),
                                                                         header=modelHeadersOn).to_numpy(),
                                                           pd.read_excel(self.modelFileFullName,
                                                                         sheet_name=self.newNN_comboBox_YTrainSource.currentText(),
                                                                         header=modelHeadersOn).to_numpy(),
                                                           featureScaling=self.newNN_comboBox_featureScale.currentText(),
                                                           avgNormalisation=self.newNN_comboBox_avgNormalise.currentText())                                                        
        
        pause = time.time()
        ### Catches errors if architecture does not fit data
        in_wrong,out_bias_wrong,out_wrong,message = False,False,False,""
        if self.layer_spinBox_nodes[0].value()-self.layer_checkBox_bias[0].isChecked() != n_train:
            self.layer_spinBox_nodes[0].setValue(n_train+self.layer_checkBox_bias[0].isChecked())
            in_wrong = True
            message += "Number of input layer_nodes wrong: Auto-corrected.\n"
        if self.layer_checkBox_bias[-1].isChecked():
            self.layer_checkBox_bias[-1].setChecked(False)
            out_bias_wrong = True
            message += "No bias unit permitted on output layer: Auto-corrected.\n"
        if self.layer_spinBox_nodes[-1].value() != K_train:
            self.layer_spinBox_nodes[-1].setValue(K_train)
            out_bias_wrong = True
            message += "Number of output layer_nodes wrong: Auto-corrected."
        if in_wrong or out_bias_wrong or out_wrong:
            msg = QMessageBox()
            msg.setWindowTitle("Errors corrected")
            msg.setText(message)
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        endpause = time.time()
        
        ### Sets up NN
        self.s = [x.value() for x in self.layer_spinBox_nodes]
        self.edges = [np.ones((self.s[i]-self.layer_checkBox_bias[i].isChecked(),self.s[i-1])) for i in range(1,len(self.s))]
        self.my_NN = NN.NN(biases = [self.layer_checkBox_bias[i].isChecked() for i in range(0,self.newNN_spinBox_noLayers.value())],
                           edges = self.edges,
                           activationKind = self.newNN_comboBox_actFunKind.currentText(),
                           avgNormalisation = avgs_train,
                           featureScaling = scales_train,
                           weightInitType = self.newNN_comboBox_weightInit.currentText())
        lambdas = [float(x) for x in self.newNN_tex_regVals.toPlainText().split(',')] if self.newNN_tex_regVals.toPlainText() != '' else [0]
        
        ### Trains it
        _,\
            best_index,\
                gradChecks,\
                    costs_graph,\
                        accuracy = self.my_NN.model(X_train,Y_train,
                                                    optimiser=self.newNN_comboBox_optimiserKind.currentText(),
                                                    lambdas=lambdas,
                                                    initLoops=self.newNN_spinBox_noInitLoops.value(),shuffleLoops=self.newNN_spinBox_noShuffleLoops.value(),
                                                    weightInitType=self.newNN_comboBox_weightInit.currentText(),
                                                    trainCVTestSplit=ar([self.newNN_spinBox_train.value(),
                                                                         self.newNN_spinBox_CV.value(),
                                                                         self.newNN_spinBox_test.value()]),
                                                    maxIters=self.newNN_spinBox_maxIters.value(),tol=self.newNN_spinBox_tol.value(),
                                                    gradDesc_alpha=self.newNN_gradDesc[1].value(),gradDesc_alphaChange=self.newNN_gradDesc[3].value(),
                                                    steepConj_maxIters=self.newNN_steepConj[1].value(),steepConj_tol=self.newNN_steepConj[3].value(),
                                                    gradCheck=disp_gradCheck,gradCheckEps=self.newNN_spinBox_gradCheckEps.value(),costsPlot=disp_costs)
        
        end_time = time.time()
        self.loading = False
        if disp_costs:
            Graph(costs_graph[1])
        self.updateNN()
        
        ### Final message box with information
        msg = QMessageBox()
        msg.setWindowTitle("Model Hyperparameters")
        msg.setWindowIcon(QIcon(self.iconName))
        msgText = "Optimisation has determined the best hyperparameters for use in your neural network.\nInitialisation, data extraction, and modelling time: "+str(end_time-endpause+pause-start_time)+"s."
        if accuracy is not None:
            msgText = msgText + "\nAccuracy from a test set (of user specified size): " + str(accuracy*100) + "%."
        msg.setText(msgText)
        msg.setIcon(QMessageBox.Information)
        if disp_gradCheck:
            msg.setStandardButtons(QMessageBox.Save|QMessageBox.Ok)
            buttonSave = msg.button(QMessageBox.Save)
            buttonSave.setText('Save Gradients')
            msg.setDefaultButton(QMessageBox.Ok)
            msg.setStyleSheet("QLabel{min-width: 400px;min-height: 200px}")
            msg.setInformativeText("Exact and approximate final gradient of the weights for chosen hyperparameters in details.")
            DetMsg = ['','']
            for j in range(2):
                DetMsg[j] = ''.join(['Layer '+str(i+1)+'-> layer '+str(i+2)+':\n'+str(gradChecks[j][i]) + '\n   ' for i in range(len(gradChecks[j]))])
            msg.setDetailedText("\nExact:\n   " + DetMsg[0] + "\nApproximation:\n   " + DetMsg[1])
            res = msg.exec_()
            if res == QMessageBox.Save:
                saved_model_Full_Name, _ = QFileDialog.getSaveFileName(self,"Save Model Details",os.getcwd(),"(*.xlsx)")
                if saved_model_Full_Name == '':
                    return
                writer = pd.ExcelWriter(saved_model_Full_Name, engine = 'xlsxwriter')
                if disp_gradCheck:
                    for exact,approx,i in zip(gradChecks[0][best_index],gradChecks[1][best_index],range(len(gradChecks[0][best_index]))):
                        pd.DataFrame(exact).to_excel(writer,sheet_name="Exact_"+str(i),index=False,header=False)
                        pd.DataFrame(approx).to_excel(writer,sheet_name="Approx_"+str(i),index=False,header=False)
                writer.save()
                writer.close()
        else:
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
    
    ### Updates current neural network section when newone modelled
    def updateNN(self):
        self.NN_lab_reg_val.setText(str(self.my_NN.Lambda))
        self.NN_lab_arcFun_val.setText(str(self.my_NN.activationKind))
        self.NN_lab_noLayersVal.setText(str(self.my_NN.noLayers))
    
    ### Predicts based on prediction data
    @errorCatcher
    def predict(self):
        self.startLoadingAnimation("Predicting")
        
        start_time = time.time()
        predHeadersOn = 0 if self.NN_checkBox_headers.isChecked() else None
        if 'predFileFullName' not in dir(self) or self.predFileFullName == '':
             raise ValueError('No file found.')
        
        X_predict,_,_,_,_,_,_ = NN.extract_data(pd.read_excel(self.predFileFullName,
                                                              sheet_name=self.NN_comboBox_XPredSource.currentText(),
                                                              header=predHeadersOn).to_numpy(),
                                                Y_on = False,
                                                featureScaling = None, avgNormalisation = None) 
        pred_res = self.my_NN.predict(X_predict,alreadyPreprocessed=False)
        
        end_time = time.time()
        self.loading = False
        
        msg = QMessageBox()
        msg.setWindowTitle("Predictions")
        msg.setWindowIcon(QIcon(self.iconName))
        msg.setText("Your neural network has predicted the results.\nPrediction time: "+str(end_time-start_time)+"s.")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Save|QMessageBox.Ok)
        buttonSave = msg.button(QMessageBox.Save)
        buttonSave.setText('Save Predictions')
        msg.setDefaultButton(QMessageBox.Ok)
        msg.setInformativeText("Prediction results in details:")
        msg.setDetailedText(str(pred_res))
        res = msg.exec_()
        if res == QMessageBox.Save:
            saved_pred_Full_Name, _ = QFileDialog.getSaveFileName(self,"Save Predictions",os.getcwd(),"(*.xlsx)")
            if saved_pred_Full_Name == '':
                return
            pd.DataFrame(pred_res).to_excel(saved_pred_Full_Name,index=False,header=False)

### Graph window if costsPlot is true
class Graph(QtWidgets.QDialog):  
    def __init__(self,fig): 
        super().__init__()
        self.setWindowTitle('Costs via Optimiser on Training Set using Best Shuffle')
        self.setWindowIcon(QIcon('NN_Icon.ico'))
        self.fig = fig
        self.canvas = FigureCanvas(self.fig) 
        self.toolbar = NavigationToolbar(self.canvas,self)
        self.setMinimumSize(700,500)
        layout = QtWidgets.QVBoxLayout() 
        layout.addWidget(self.toolbar) 
        layout.addWidget(self.canvas)
        self.setLayout(layout) 
        self.canvas.draw()
        self.show()
 
### Text windows for uses like displaying feature scaling, average normalisation and instructions
class textWindow(QMainWindow):
    def __init__(self,title,text):
        super().__init__()
        self.setObjectName("MainWindow_text")
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon('NN_Icon.ico'))
        self.setStyleSheet("font: 12pt 'Perpetua'")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setText(text)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignTop)
        self.verticalLayout.addWidget(self.label)
        self.show()

### Window displaying current neural network architecture        
class architectureWindow(QMainWindow):
    def __init__(self,NN):
        super().__init__()
        self.setObjectName("MainWindow_arch")
        self.setWindowTitle('Architecture')
        self.setWindowIcon(QIcon('NN_Icon.ico'))
        self.setStyleSheet("font: 12pt 'Perpetua'")
        
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)
        
        self.horizontalLayoutSuper = QHBoxLayout(self.centralwidget)
        self.horizontalLayoutSuper.setObjectName("horizontalLayoutSuper")
        
        self.main_scrollArea = QScrollArea(self.centralwidget)
        self.main_scrollArea.setWidgetResizable(True)
        self.main_scrollArea.setObjectName("main_scrollArea")
        self.main_scrollArea_widgetContents = QWidget()
        self.main_scrollArea_widgetContents.setObjectName("main_scrollArea_widgetContents")
        self.main_scrollArea.setWidget(self.main_scrollArea_widgetContents)

        self.horizontalLayout = QHBoxLayout(self.main_scrollArea_widgetContents)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.layer_node_bias = []
        self.layer_nodes = []
        self.layer_scrollArea = []
        self.layer_scrollArea_widgetContents = []
        self.layer_verticalLayouts = []
        for l in range(NN.noLayers):
            self.layer_scrollArea.append(QScrollArea(self.main_scrollArea_widgetContents)) 
            self.layer_scrollArea[l].setWidgetResizable(True)
            self.layer_scrollArea[l].setMinimumWidth(149)
            self.layer_scrollArea_widgetContents.append(QWidget())
            self.layer_verticalLayouts.append(QtWidgets.QVBoxLayout(self.layer_scrollArea_widgetContents[l]))
            self.layer_verticalLayouts[l].setAlignment(Qt.AlignTop|Qt.AlignHCenter)
            
            self.layer_node_bias.append(QLabel(self.layer_scrollArea_widgetContents[l]))
            self.layer_node_bias[l].setText('0')
            self.layer_node_bias[l].setStyleSheet("border: 8px solid #FF88FF; border-radius: 50px;")
            self.layer_node_bias[l].setFixedSize(100,100)
            self.layer_node_bias[l].setAlignment(Qt.AlignCenter)
            self.layer_node_bias[l].setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
            node_bias_retainSize = self.layer_node_bias[l].sizePolicy()
            node_bias_retainSize.setRetainSizeWhenHidden(True)
            self.layer_node_bias[l].setSizePolicy(node_bias_retainSize)
            if NN.biases[l]:
                self.layer_node_bias[l].show()
            else:
                self.layer_node_bias[l].hide()    
            self.layer_verticalLayouts[l].addWidget(self.layer_node_bias[l])
            
            self.layer_nodes.append([])
            for n in range(NN[l].no_nodes-NN.biases[l]):
                self.layer_nodes[l].append(QLabel(self.layer_scrollArea_widgetContents[l]))
                self.layer_nodes[l][n].setText(str(n+1))
                self.layer_nodes[l][n].setStyleSheet("border: 8px solid #8888FF; border-radius: 50px;")
                self.layer_nodes[l][n].setFixedSize(100,100)
                self.layer_nodes[l][n].setAlignment(Qt.AlignCenter)
                self.layer_verticalLayouts[l].addWidget(self.layer_nodes[l][n])
            
            self.layer_scrollArea[l].setWidget(self.layer_scrollArea_widgetContents[l])
            self.horizontalLayout.addWidget(self.layer_scrollArea[l])
            self.horizontalLayoutSuper.addWidget(self.main_scrollArea)
        self.resize(min(NN.noLayers*150+200,1500),min(max([len(n) for n in self.layer_nodes])*100+200,800))
        self.show()
        
if __name__ == "__main__":
    print('<program>  Copyright (C) <year>  <name of author>\
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w\'.\
    This is free software, and you are welcome to redistribute it\
    under certain conditions; type `show c\' for details.')
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    MainWindow = MainWindow()
    MainWindow.show()    
    sys.exit(app.exec_())