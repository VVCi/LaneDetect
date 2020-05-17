#-------------------------------------------------
#
# Project created by QtCreator 2016-11-07T01:37:56
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Lane
TEMPLATE = app
INCLUDEPATH += /usr/local/include/opencv
LIBS += /usr/local/lib/*.so

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui
