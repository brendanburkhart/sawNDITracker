/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  Author(s):  Anton Deguet
  Created on: 2014-07-21

  (C) Copyright 2014-2017 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsNDISerialControllerQtWidget_h
#define _mtsNDISerialControllerQtWidget_h

#include <cisstMultiTask/mtsComponent.h>
#include <cisstParameterTypes/prmPositionCartesianGet.h>
#include <cisstParameterTypes/prmStateJoint.h>
#include <cisstVector/vctQtWidgetFrame.h>
#include <cisstMultiTask/mtsIntervalStatisticsQtWidget.h>
#include <cisstMultiTask/mtsMessageQtWidget.h>

#include <QWidget>
#include <QCheckBox>
#include <QSpinBox>

// Always include last
#include <sawNDITracker/sawNDITrackerQtExport.h>

class CISST_EXPORT mtsNDISerialControllerQtWidget : public QWidget, public mtsComponent
{
    Q_OBJECT;
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION_ONEARG, CMN_LOG_ALLOW_DEFAULT);

public:
    mtsNDISerialControllerQtWidget(const std::string & componentName, double periodInSeconds = 50.0 * cmn_ms);
    ~mtsNDISerialControllerQtWidget() {}

    void Configure(const std::string & filename = "");
    void Startup(void);
    void Cleanup(void);

protected:
    virtual void closeEvent(QCloseEvent * event);

private slots:
    void timerEvent(QTimerEvent * event);

private:
    //! setup GUI
    void setupUi(void);
    int TimerPeriodInMilliseconds;

protected:
    struct {
        mtsFunctionRead  GetPeriodStatistics;
        mtsFunctionWrite Connect;
        mtsFunctionVoid  Disconnect;
        mtsFunctionWrite Track;
        mtsFunctionWrite Beep;
    } Tracker;

private:
    // Control
    QCheckBox * QCBConnect;
    QCheckBox * QCBTrack;
    QSpinBox * QSBBeepCount;

    // Timing
    mtsIntervalStatistics IntervalStatistics;
    mtsIntervalStatisticsQtWidget * QMIntervalStatistics;

    // Messages
    mtsMessageQtWidget * QMMessage;

 private slots:
    void SlotConnect(bool);
    void SlotTrack(bool);
    void SlotBeep(void);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsNDISerialControllerQtWidget);

#endif // _mtsNDISerialControllerQtWidget_h
