/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  Author(s):  Anton Deguet, Ali Uneri
  Created on: 2009-10-13

  (C) Copyright 2009-2017 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <bitset>

#include <cisstCommon/cmnStrings.h>
#include <cisstVector/vctDynamicMatrixTypes.h>

#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <sawNDITracker/mtsNDISerial.h>

CMN_IMPLEMENT_SERVICES_DERIVED_ONEARG(mtsNDISerial, mtsTaskPeriodic, mtsTaskPeriodicConstructorArg);


#if (CISST_OS == CISST_LINUX) || (CISST_OS == CISST_DARWIN)
#include <glob.h>
inline bool Glob(const std::string & pattern, std::vector<std::string> & paths) {
    glob_t glob_result;
    bool result = glob(pattern.c_str(), 0, 0, &glob_result);
    for (unsigned int i = 0; i < glob_result.gl_pathc; i++) {
        paths.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return result;
}
#endif


void mtsNDISerial::Init(void)
{
    mReadTimeout = 2.0 * cmn_s;
    mIsTracking = false;
    mTrackStrayMarkers = true;
    mStrayMarkers.SetSize(50, 5);
    mStrayMarkers.Zeros();
    memset(mSerialBuffer, 0, MAX_BUFFER_SIZE);
    mSerialBufferPointer = mSerialBuffer;

    // default search path to locate roms files / tool definitions
    mDefinitionPath.Add(cmnPath::GetWorkingDirectory());
    mDefinitionPath.Add(std::string(sawNDITracker_SOURCE_DIR) + "/../share/roms", cmnPath::TAIL);

    mConfigurationStateTable = new mtsStateTable(100, "Configuration");
    mConfigurationStateTable->SetAutomaticAdvance(false);
    this->AddStateTable(mConfigurationStateTable);
    mConfigurationStateTable->AddData(this->Name, "TrackerName");
    mConfigurationStateTable->AddData(mSerialPortName, "SerialPort");
    mConfigurationStateTable->AddData(mToolNames, "ToolNames");

    StateTable.AddData(mIsTracking, "IsTracking");
    StateTable.AddData(mTrackStrayMarkers, "TrackStrayMarkers");
    StateTable.AddData(mStrayMarkers, "StrayMarkers");

    mControllerInterface = AddInterfaceProvided("Controller");
    if (mControllerInterface) {
        mControllerInterface->AddMessageEvents();
        mControllerInterface->AddCommandWrite(&mtsNDISerial::Connect, this, "Connect");
        mControllerInterface->AddCommandVoid(&mtsNDISerial::Disconnect, this, "Disconnect");
        mControllerInterface->AddCommandWrite(&mtsNDISerial::Beep, this, "Beep");
        mControllerInterface->AddCommandVoid(&mtsNDISerial::PortHandlesInitialize, this, "PortHandlesInitialize");
        mControllerInterface->AddCommandVoid(&mtsNDISerial::PortHandlesQuery, this, "PortHandlesQuery");
        mControllerInterface->AddCommandVoid(&mtsNDISerial::PortHandlesEnable, this, "PortHandlesEnable");
        mControllerInterface->AddCommandVoid(&mtsNDISerial::ReportStrayMarkers, this, "ReportStrayMarkers");
        mControllerInterface->AddCommandWrite(&mtsNDISerial::ToggleTracking, this, "ToggleTracking");
        mControllerInterface->AddCommandReadState(*mConfigurationStateTable, this->Name, "Name");
        mControllerInterface->AddCommandReadState(*mConfigurationStateTable, mSerialPortName, "SerialPort");
        mControllerInterface->AddCommandReadState(*mConfigurationStateTable, mToolNames, "ToolNames");
        mControllerInterface->AddCommandReadState(StateTable, mIsTracking, "IsTracking");
        mControllerInterface->AddCommandReadState(StateTable, mTrackStrayMarkers, "TrackStrayMarkers");
        mControllerInterface->AddCommandReadState(StateTable, mStrayMarkers, "StrayMarkers");
        mControllerInterface->AddCommandReadState(StateTable, StateTable.PeriodStats,
                                                  "GetPeriodStatistics");
        mControllerInterface->AddEventWrite(Events.Connected, "Connected", std::string(""));
        mControllerInterface->AddEventWrite(Events.Tracking, "Tracking", false);
        mControllerInterface->AddEventVoid(Events.UpdatedTools, "UpdatedTools");
    }

    mConfigurationStateTable->Start();
    mConfigurationStateTable->Advance();
}


void mtsNDISerial::SetSerialPort(const std::string & serialPort)
{
    mSerialPortName = serialPort;
}


void mtsNDISerial::Configure(const std::string & filename)
{
    CMN_LOG_CLASS_INIT_VERBOSE << "Configure: using " << filename << std::endl;

    if (filename.empty()) {
        return;
    }

    std::ifstream jsonStream;
    jsonStream.open(filename.c_str());

    Json::Value jsonConfig, jsonValue;
    Json::Reader jsonReader;
    // make sure the file valid json
    if (!jsonReader.parse(jsonStream, jsonConfig)) {
        CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to parse configuration" << std::endl
                                 << "File: " << filename << std::endl << "Error(s):" << std::endl
                                 << jsonReader.getFormattedErrorMessages();
        return;
    }
    // keep the content of the file in cisstLog for debugging
    CMN_LOG_CLASS_INIT_VERBOSE << "Configure: " << this->GetName()
                               << " using file \"" << filename << "\"" << std::endl
                               << "----> content of configuration file: " << std::endl
                               << jsonConfig << std::endl
                               << "<----" << std::endl;


    // start looking for configuration parameters
    jsonValue = jsonConfig["serial-port"];
    // if the port is specified in the json file
    if (!jsonValue.empty()) {
        // and if it has not already been set
        if (mSerialPortName == "") {
            mSerialPortName = jsonValue.asString();
            if (mSerialPortName == "") {
                CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to convert \"serial-port\" to a string" << std::endl;
                return;
            }
            CMN_LOG_CLASS_INIT_VERBOSE << "Configure: found \"serial-port\": " << mSerialPortName << std::endl;
        } else {
            CMN_LOG_CLASS_INIT_WARNING << "Configure: \"serial-port\" in file \"" << filename
                                       << "\" will be ignored since the serial port has already been set as: "
                                       << mSerialPortName << std::endl;
        }
    }

    // path to locate tool definitions
    const Json::Value definitionPath = jsonConfig["definition-path"];
    // preserve order from config file
    for (int index = (definitionPath.size() - 1);
         index >= 0;
         --index) {
        std::string path = definitionPath[index].asString();
        if (path != "") {
            mDefinitionPath.Add(path, cmnPath::HEAD);
        }
    }

    // get tools defined by user
    const Json::Value jsonTools = jsonConfig["tools"];
    for (unsigned int index = 0; index < jsonTools.size(); ++index) {
        std::string name, serialNumber, definition;
        const Json::Value jsonTool = jsonTools[index];
        jsonValue = jsonTool["name"];
        if (!jsonValue.empty()) {
            name = jsonValue.asString();
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to find \"name\" for tools["
                                     << index << "]" << std::endl;
            return;
        }
        jsonValue = jsonTool["serial-number"];
        if (!jsonValue.empty()) {
            serialNumber = jsonValue.asString();
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "Configure: failed to find \"serial-number\" for tools["
                                     << index << "]" << std::endl;
            return;
        }
        jsonValue = jsonTool["definition"];
        if (!jsonValue.empty()) {
            definition = jsonValue.asString();
            // try to locate the file
            if (!cmnPath::Exists(definition)) {
                CMN_LOG_CLASS_INIT_VERBOSE << "Configure: definition file \"" << definition
                                           << "\" not found, using definition-paths to locate it."
                                           << std::endl;
                std::string fullPath = mDefinitionPath.Find(definition);
                if (fullPath != "") {
                    CMN_LOG_CLASS_INIT_VERBOSE << "Configure: found definition file \"" << fullPath
                                               << "\" for \"" << definition << "\"" << std::endl;
                    definition = fullPath;
                } else {
                    CMN_LOG_CLASS_INIT_ERROR << "Configure: can't find definition file \"" << definition
                                             << "\" using search path: " << mDefinitionPath << std::endl;
                    return;
                }
            }
        } else {
            definition = "";
        }

        AddTool(name, serialNumber, definition);
    }

#if 0
            context << "/tooltip";
            std::string rotation, translation;
            config.GetXMLValue(context.str().c_str(), "@rotation", rotation);
            config.GetXMLValue(context.str().c_str(), "@translation", translation);
            if (!rotation.empty()) {
                CMN_LOG_CLASS_INIT_ERROR << "Configure: tooltip rotation will not be applied (not implemented)" << std::endl;
            }
            if (!translation.empty()) {
                std::stringstream offset(translation);
                double value;
                for (unsigned int j = 0; offset >> value; j++) {
                    tool->TooltipOffset[j] = value;
                    offset.ignore(1);
                }
            }
        }
#endif
}

void mtsNDISerial::Connect(const std::string & serialPortName)
{
    // in case someone calls Connect multiple times
    if (mSerialPort.IsOpened()) {
        mSerialPort.Close();
        mControllerInterface->SendStatus(this->GetName() + ": serial port was opened, closing first");
    }

    // if this method is called with a serial port, overwrite the existing one
    if (!serialPortName.empty()) {
        mSerialPortName = serialPortName;
    }

    // first try to connect using a port name provided either in the
    // config file or using a command line argument and the method
    // SetSerialPort
    if (!mSerialPortName.empty()) {
        mSerialPort.SetPortName(mSerialPortName);
        if (!mSerialPort.Open()) {
            mControllerInterface->SendError(this->GetName() + ": failed to open serial port: "
                                            + mSerialPort.GetPortName());
            return;
        }
        mControllerInterface->SendStatus(this->GetName() + ": found serial port: "
                                         + mSerialPort.GetPortName());
        // now try to reset port
        if (!ResetSerialPort()) {
            mControllerInterface->SendError(this->GetName() + ": failed to reset serial port: "
                                            + mSerialPort.GetPortName());
            mSerialPort.Close();
            return;
        }

    } else {
        // this is a bit brutal, build a list of possible names
        // depending on the OS and try them all one by one
        mControllerInterface->SendWarning(this->GetName() + ": no serial port specified, trying to discover automatically");
        std::vector<std::string> ports;
#if (CISST_OS == CISST_WINDOWS)
        for (size_t i = 1; i <= 256; i++) {
            std::ostringstream stream;
            stream << "COM" << i;
            ports.push_back(stream.str());
        }
#elif (CISST_OS == CISST_LINUX)
        Glob("/dev/ttyS*", ports);
        Glob("/dev/ttyUSB*", ports);
#elif (CISST_OS == CISST_DARWIN)
        Glob("/dev/tty*", ports);
        Glob("/dev/cu*", ports);
#endif
        for (size_t i = 0; i < ports.size(); i++) {
            mSerialPort.SetPortName(ports[i]);
            mControllerInterface->SendStatus(this->GetName() + ": trying to open serial port: "
                                             + mSerialPort.GetPortName());
            // try to reset only if we can open
            if (mSerialPort.Open()) {
                mControllerInterface->SendStatus(this->GetName() + ": trying to reset serial port: "
                                                 + mSerialPort.GetPortName());
                if (ResetSerialPort()) {
                    // looks like we found it!
                    mSerialPortName = mSerialPort.GetPortName();
                    break;
                }
                mSerialPort.Close();
            }
        }
    }

    // if we get here we've been able to reset the device
    mControllerInterface->SendStatus(this->GetName() + ": device found on port: "
                                     + mSerialPort.GetPortName());

    // increase the timeout during initialization
    const double previousTimeout = mReadTimeout;
    mReadTimeout = 5.0 * cmn_s;

    SetSerialPortSettings(osaSerialPort::BaudRate115200,
                          osaSerialPort::CharacterSize8,
                          osaSerialPort::ParityCheckingNone,
                          osaSerialPort::StopBitsOne,
                          osaSerialPort::FlowControlNone);

    // initialize NDI controller
    CommandSend("INIT ");
    if (ResponseRead("OKAY")) {
        mControllerInterface->SendStatus(this->GetName() + ": device initialized");
    } else {
        mControllerInterface->SendError(this->GetName() + ": device failed to initialize");
        mSerialPort.Close();
        mReadTimeout = previousTimeout;
        return;
    }

    // now we're connected
    mConfigurationStateTable->Start();
    mConfigurationStateTable->Advance();
    Events.Connected(mSerialPortName);

    // get some extra information on the system for debug/log
    CommandSend("VER 0");
    ResponseRead();
    mControllerInterface->SendStatus(this->GetName() + ": command VER 0 returned:\n" + mSerialBuffer);

    CommandSend("VER 3");
    ResponseRead();
    mControllerInterface->SendStatus(this->GetName() + ": command VER 3 returned:\n" + mSerialBuffer);

    CommandSend("VER 4");
    ResponseRead();
    mControllerInterface->SendStatus(this->GetName() + ": command VER 4 returned:\n" + mSerialBuffer);

    CommandSend("VER 5");
    if (ResponseRead("024")) {
        mControllerInterface->SendStatus(this->GetName() + ": device firmware is 024 (supported)");
    } else {
        mControllerInterface->SendError(this->GetName() + ": device firmware is not what we're expecting, got: "
                                        + mSerialBuffer);
        mReadTimeout = previousTimeout;
        return;
    }

    mReadTimeout = previousTimeout;

    PortHandlesInitialize();

    PortHandlesPassiveTools();

    PortHandlesQuery();

    PortHandlesEnable();
}

void mtsNDISerial::Disconnect(void)
{
    // just in case we were tracking
    ToggleTracking(false);
    // if we can't properly stop tracking, still set the flag
    mIsTracking = false;
    // close serial port
    mSerialPort.Close();
    // send event
    Events.Connected(std::string(""));
    mControllerInterface->SendStatus(this->GetName() + ": serial port disconnected");
}

void mtsNDISerial::Run(void)
{
    ProcessQueuedCommands();
    if (mIsTracking) {
        Track();
    }
}


void mtsNDISerial::Cleanup(void)
{
    ToggleTracking(false);
    if (!mSerialPort.Close()) {
        CMN_LOG_CLASS_INIT_ERROR << "Cleanup: failed to close serial port" << std::endl;
    }
}


void mtsNDISerial::CommandInitialize(void)
{
    mSerialBufferPointer = mSerialBuffer;
}


void mtsNDISerial::CommandAppend(const char command)
{
    *mSerialBufferPointer = command;
    mSerialBufferPointer++;
}


void mtsNDISerial::CommandAppend(const char * command)
{
    const size_t size = strlen(command);
    strncpy(mSerialBufferPointer, command, size);
    mSerialBufferPointer += size;
}


void mtsNDISerial::CommandAppend(const int command)
{
    mSerialBufferPointer += cmn_snprintf(mSerialBufferPointer, GetSerialBufferAvailableSize(), "%d", command);
}


unsigned int mtsNDISerial::ComputeCRC(const char * data)
{
    static unsigned char oddParity[16] = { 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 };
    unsigned char * dataPointer = (unsigned char *)data;
    unsigned int temp = 0;
    unsigned int crc = 0;

    while (*dataPointer) {
        temp = (*dataPointer ^ (crc & 0xff)) & 0xff;
        crc >>= 8;

        if (oddParity[temp & 0x0f] ^ oddParity[temp >> 4]) {
            crc ^= 0xc001;
        }
        temp <<= 6;
        crc ^= temp;
        temp <<= 1;
        crc ^= temp;
        dataPointer++;
    }
    return crc;
}


bool mtsNDISerial::CommandSend(void)
{
    CommandAppend('\r');
    CommandAppend('\0');

    const int bytesToSend = static_cast<int>(strlen(mSerialBuffer));
    const int bytesSent = mSerialPort.Write(mSerialBuffer, bytesToSend);
    if (bytesSent != bytesToSend) {
        CMN_LOG_CLASS_RUN_ERROR << "SendCommand: sent only " << bytesSent << " of " << bytesToSend
                                << " for command \"" << mSerialBuffer << "\"" << std::endl;
        return false;
    }
    CMN_LOG_CLASS_RUN_DEBUG << "SendCommand: successfully sent command \""
                            << mSerialBuffer << "\"" << std::endl;
    return true;
}


bool mtsNDISerial::ResponseRead(void)
{
    mResponseTimer.Reset();
    mResponseTimer.Start();

    mSerialBufferPointer = mSerialBuffer;

    bool receivedMessage = false;
    do {
        int bytesRead = mSerialPort.Read(mSerialBufferPointer,
                                         static_cast<int>(GetSerialBufferAvailableSize()));
        if (bytesRead > 0) {
            mSerialBufferPointer += bytesRead;
            receivedMessage = (GetSerialBufferSize() > 0)
                && (*(mSerialBufferPointer - 1) == '\r');
        }
    } while ((mResponseTimer.GetElapsedTime() < mReadTimeout)
             && !receivedMessage);

    mResponseTimer.Stop();

    if (mResponseTimer.GetElapsedTime() > mReadTimeout) {
        CMN_LOG_CLASS_RUN_ERROR << "ResponseRead: read command timed out (timeout is "
                                << mReadTimeout << "s)" << std::endl;
        return false;
    }

    if (!ResponseCheckCRC()) {
        return false;
    }
    return true;
}


bool mtsNDISerial::ResponseRead(const char * expectedMessage)
{
    if (!ResponseRead()) {
        CMN_LOG_CLASS_RUN_ERROR << "ResponseRead: timeout while waiting for \"" << expectedMessage << "\"" << std::endl;
        return false;
    }

    if (strncmp(expectedMessage, mSerialBuffer, GetSerialBufferStringSize()) != 0) {
        CMN_LOG_CLASS_RUN_ERROR << "ResponseRead: expected \"" << expectedMessage
                                << "\", but received \"" << mSerialBuffer << "\"" << std::endl;
        return false;
    }
    CMN_LOG_CLASS_RUN_DEBUG << "ResponseRead: received expected response" << std::endl;
    return true;
}


bool mtsNDISerial::ResponseCheckCRC(void)
{
    char receivedCRC[CRC_SIZE + 1];
    char computedCRC[CRC_SIZE + 1];
    char * crcPointer = mSerialBufferPointer - (CRC_SIZE + 1);  // +1 for '\r'

    // extract CRC from buffer
    strncpy(receivedCRC, crcPointer, CRC_SIZE);
    receivedCRC[CRC_SIZE] = '\0';
    *crcPointer = '\0';
    mSerialBufferPointer = crcPointer + 1;

    // compute CRC
    sprintf(computedCRC, "%04X", ComputeCRC(mSerialBuffer));
    computedCRC[CRC_SIZE] = '\0';

    // compare CRCs
    if (strncmp(receivedCRC, computedCRC, CRC_SIZE) != 0) {
        CMN_LOG_CLASS_RUN_ERROR << "ResponseCheckCRC: received \"" << mSerialBuffer << receivedCRC
                                << "\", but computed \"" << computedCRC << "\" for CRC" << std::endl;
        return false;
    }
    CMN_LOG_CLASS_RUN_DEBUG << "ResponseCheckCRC: CRC check was successful for \"" << mSerialBuffer << "\"" << std::endl;
    return true;
}


bool mtsNDISerial::ResetSerialPort(void)
{
    mSerialPort.SetBaudRate(osaSerialPort::BaudRate9600);
    mSerialPort.SetCharacterSize(osaSerialPort::CharacterSize8);
    mSerialPort.SetParityChecking(osaSerialPort::ParityCheckingNone);
    mSerialPort.SetStopBits(osaSerialPort::StopBitsOne);
    mSerialPort.SetFlowControl(osaSerialPort::FlowControlNone);
    mSerialPort.Configure();

    const double breakTime = 0.5 * cmn_s;
    mSerialPort.WriteBreak(breakTime);
    // wait for length of break and a bit more
    Sleep(breakTime + 0.5 * cmn_s);

    // temporary increase timeout to leave time for the system to boot
    const double previousReadTimeout = mReadTimeout;
    mReadTimeout = 5.0 * cmn_s;
    if (!ResponseRead("RESET")) {
        CMN_LOG_CLASS_INIT_ERROR << "ResetSerialPort: failed to reset" << std::endl;
        mReadTimeout = previousReadTimeout;
        return false;
    }
    mReadTimeout = previousReadTimeout;
    return true;
}


bool mtsNDISerial::SetSerialPortSettings(osaSerialPort::BaudRateType baudRate,
                                         osaSerialPort::CharacterSizeType characterSize,
                                         osaSerialPort::ParityCheckingType parityChecking,
                                         osaSerialPort::StopBitsType stopBits,
                                         osaSerialPort::FlowControlType flowControl)
{
    CommandInitialize();
    CommandAppend("COMM ");

    switch (baudRate) {
        case osaSerialPort::BaudRate9600:
            CommandAppend('0');
            break;
        case osaSerialPort::BaudRate19200:
            CommandAppend('2');
            break;
        case osaSerialPort::BaudRate38400:
            CommandAppend('3');
            break;
        case osaSerialPort::BaudRate57600:
            CommandAppend('4');
            break;
        case osaSerialPort::BaudRate115200:
            CommandAppend('5');
            break;
        default:
            CMN_LOG_CLASS_INIT_ERROR << "SetSerialPortSettings: invalid baud rate" << std::endl;
            return false;
            break;
    }

    switch (characterSize) {
        case osaSerialPort::CharacterSize8:
            CommandAppend('0');
            break;
        case osaSerialPort::CharacterSize7:
            CommandAppend('1');
            break;
        default:
            CMN_LOG_CLASS_INIT_ERROR << "SetSerialPortSettings: invalid character size" << std::endl;
            return false;
            break;
    }

    switch (parityChecking) {
        case osaSerialPort::ParityCheckingNone:
            CommandAppend('0');
            break;
        case osaSerialPort::ParityCheckingOdd:
            CommandAppend('1');
            break;
        case osaSerialPort::ParityCheckingEven:
            CommandAppend('2');
            break;
        default:
            CMN_LOG_CLASS_INIT_ERROR << "SetSerialPortSettings: invalid parity checking" << std::endl;
            return false;
            break;
    }

    switch (stopBits) {
        case osaSerialPort::StopBitsOne:
            CommandAppend('0');
            break;
        case osaSerialPort::StopBitsTwo:
            CommandAppend('1');
            break;
        default:
            CMN_LOG_CLASS_INIT_ERROR << "SetSerialPortSettings: invalid stop bits" << std::endl;
            return false;
            break;
    }

    switch (flowControl) {
        case osaSerialPort::FlowControlNone:
            CommandAppend('0');
            break;
        case osaSerialPort::FlowControlHardware:
            CommandAppend('1');
            break;
        default:
            CMN_LOG_CLASS_INIT_ERROR << "SetSerialPortSettings: invalid flow control" << std::endl;
            return false;
            break;
    }

    if (!CommandSend()) {
        mControllerInterface->SendError(this->GetName() + ": SetSerialPortSettings: failed to send command");
        return false;
    }

    if (ResponseRead("OKAY")) {
        Sleep(200.0 * cmn_ms);
        mSerialPort.SetBaudRate(baudRate);
        mSerialPort.SetCharacterSize(characterSize);
        mSerialPort.SetParityChecking(parityChecking);
        mSerialPort.SetStopBits(stopBits);
        mSerialPort.SetFlowControl(flowControl);
        mSerialPort.Configure();
        Sleep(200.0 * cmn_ms);
        mControllerInterface->SendStatus(this->GetName() + ": SetSerialPortSettings succeeded");
        return true;
    }
    mControllerInterface->SendError(this->GetName() + ": SetSerialPortSettings: didn't receive \"OKAY\"");
    return false;
}


void mtsNDISerial::Beep(const int & numberOfBeeps)
{
    if (numberOfBeeps < 1 || numberOfBeeps > 9) {
        CMN_LOG_CLASS_RUN_ERROR << "Beep: invalid input: " << numberOfBeeps << ", must be between 0-9" << std::endl;
    }
    CMN_LOG_CLASS_RUN_VERBOSE << "Beep: beeing " << numberOfBeeps << " times" << std::endl;
    do {
        CommandInitialize();
        CommandAppend("BEEP ");
        CommandAppend(numberOfBeeps);
        CommandSend();
        Sleep(100.0 * cmn_ms);
        if (!ResponseRead()) {
            return;
        }
    } while (strncmp("0", mSerialBuffer, 1) == 0);

    if (strncmp("1", mSerialBuffer, 1) != 0) {
        CMN_LOG_CLASS_RUN_ERROR << "Beep: unknown response received: "
                                << mSerialBuffer << std::endl;
    }
}


void mtsNDISerial::LoadToolDefinitionFile(const char * portHandle,
                                          const std::string & filePath)
{
    std::ifstream toolDefinitionFile(filePath.c_str(), std::ios::binary);
    if (!toolDefinitionFile.is_open()) {
        CMN_LOG_CLASS_INIT_ERROR << "LoadToolDefinitionFile: could not open " << filePath << std::endl;
        return;
    }

    toolDefinitionFile.seekg(0, std::ios::end);
    size_t fileSize = toolDefinitionFile.tellg();
    size_t definitionSize = fileSize * 2;
    size_t paddingSize = 128 - (definitionSize % 128);
    size_t numChunks = (definitionSize + paddingSize) / 128;
    toolDefinitionFile.seekg(0, std::ios::beg);

    if (fileSize > 960) {
        CMN_LOG_CLASS_INIT_ERROR << "LoadToolDefinitionFile: " << filePath << " of size "
                                 << fileSize << " bytes exceeds the 960 bytes limit" << std::endl;
        return;
    }

    char input[65] = { 0 };
    input[64] = '\0';
    char output[129];
    output[128] = '\0';
    char address[5];
    address[4] = '\0';

    for (unsigned int i = 0; i < numChunks; i++) {
        toolDefinitionFile.read(input, 64);
        for (unsigned int j = 0; j < 64; j++) {
            sprintf(&output[j*2], "%02X", static_cast<unsigned char>(input[j]));
        }
        sprintf(address, "%04X", i * 64);
        CommandInitialize();
        CommandAppend("PVWR ");
        CommandAppend(portHandle);
        CommandAppend(address);
        CommandAppend(output);
        CommandSend();
        ResponseRead("OKAY");
    }
}


mtsNDISerial::Tool * mtsNDISerial::CheckTool(const std::string & serialNumber)
{
    const ToolsType::const_iterator end = mTools.end();
    ToolsType::const_iterator toolIterator;
    for (toolIterator = mTools.begin();
         toolIterator != end;
         ++toolIterator) {
        if (toolIterator->second->SerialNumber == serialNumber) {
            CMN_LOG_CLASS_INIT_DEBUG << "CheckTool: found existing tool for serial number: " << serialNumber << std::endl;
            return toolIterator->second;
        }
    }
    return 0;
}


mtsNDISerial::Tool * mtsNDISerial::AddTool(const std::string & name,
                                           const std::string & serialNumber,
                                           const std::string & toolDefinitionFile)
{
    Tool * tool = CheckTool(serialNumber);

    if (tool) {
        CMN_LOG_CLASS_INIT_WARNING << "AddTool: there's already a tool with serial number \"" << serialNumber
                                   << "\", name: " << name << ".  Ignoring request to add tool" << std::endl;
        return tool;
    }

    tool = new Tool();
    tool->Name = name;
    tool->SerialNumber = serialNumber;
    tool->Definition = toolDefinitionFile;

    if (!mTools.AddItem(tool->Name, tool, CMN_LOG_LEVEL_INIT_ERROR)) {
        CMN_LOG_CLASS_INIT_ERROR << "AddTool: no tool created, duplicate name exists: " << name << std::endl;
        delete tool;
        return 0;
    }
    CMN_LOG_CLASS_INIT_VERBOSE << "AddTool: created tool \"" << name << "\" with serial number: " << serialNumber << std::endl;

    // create an interface for tool
    tool->Interface = AddInterfaceProvided(name);
    if (tool->Interface) {
        tool->Interface->AddCommandRead(&mtsStateTable::GetIndexReader, &StateTable, "GetTableIndex");
        StateTable.AddData(tool->TooltipPosition, name + "Position");
        tool->Interface->AddCommandReadState(StateTable, tool->TooltipPosition, "GetPositionCartesian");
        StateTable.AddData(tool->MarkerPosition, name + "Marker");
        tool->Interface->AddCommandReadState(StateTable, tool->MarkerPosition, "GetMarkerCartesian");
    }

    // update list of existing tools
    mConfigurationStateTable->Start(); {
        mToolNames = mTools.GetNames();
    } mConfigurationStateTable->Advance();
    Events.UpdatedTools();

    return tool;
}


std::string mtsNDISerial::GetToolName(const size_t index) const
{
    ToolsType::const_iterator toolIterator = mTools.begin();
    if (index >= mTools.size()) {
        CMN_LOG_CLASS_RUN_ERROR << "GetToolName: requested index is out of range" << std::endl;
        return "";
    }
    for (size_t i = 0; i < index; i++) {
        toolIterator++;
    }
    return toolIterator->first;
}


void mtsNDISerial::PortHandlesInitialize(void)
{
    char * parsePointer;
    unsigned int numPortHandles = 0;
    std::vector<vctChar3> portHandles;

    // are there port handles to be freed?
    CommandSend("PHSR 01");
    ResponseRead();
    parsePointer = mSerialBuffer;
    sscanf(parsePointer, "%02X", &numPortHandles);
    parsePointer += 2;
    portHandles.resize(numPortHandles);
    for (unsigned int i = 0; i < portHandles.size(); i++) {
        sscanf(parsePointer, "%2c%*3c", portHandles[i].Pointer());
        parsePointer += 5;
        portHandles[i][2] = '\0';
    }
    for (unsigned int i = 0; i < portHandles.size(); i++) {
        CommandInitialize();
        CommandAppend("PHF ");
        CommandAppend(portHandles[i].Pointer());
        CommandSend();
        ResponseRead("OKAY");
        CMN_LOG_CLASS_RUN_DEBUG << "PortHandlesInitialize: freed port handle: " << portHandles[i].Pointer() << std::endl;
    }

    // are there port handles to be initialized?
    CommandSend("PHSR 02");
    ResponseRead();
    parsePointer = mSerialBuffer;
    sscanf(parsePointer, "%02X", &numPortHandles);
    parsePointer += 2;
    portHandles.resize(numPortHandles);
    for (unsigned int i = 0; i < portHandles.size(); i++) {
        sscanf(parsePointer, "%2c%*3c", portHandles[i].Pointer());
        parsePointer += 5;
        portHandles[i][2] = '\0';
    }
    for (unsigned int i = 0; i < portHandles.size(); i++) {
        CommandInitialize();
        CommandAppend("PINIT ");
        CommandAppend(portHandles[i].Pointer());
        CommandSend();
        ResponseRead("OKAY");
        CMN_LOG_CLASS_RUN_DEBUG << "PortHandlesInitialize: initialized port handle: " << portHandles[i].Pointer() << std::endl;
    }
}


void mtsNDISerial::PortHandlesQuery(void)
{
    char * parsePointer;
    unsigned int numPortHandles = 0;
    std::vector<vctChar3> portHandles;

    CommandSend("PHSR 00");
    ResponseRead();
    parsePointer = mSerialBuffer;
    sscanf(parsePointer, "%02X", &numPortHandles);
    parsePointer += 2;
    CMN_LOG_CLASS_INIT_DEBUG << "PortHandlesQuery: " << numPortHandles << " tools are plugged in" << std::endl;
    portHandles.resize(numPortHandles);
    for (unsigned int i = 0; i < portHandles.size(); i++) {
        sscanf(parsePointer, "%2c%*3c", portHandles[i].Pointer());
        parsePointer += 5;
        portHandles[i][2] = '\0';
    }

    Tool * tool;
    std::string toolKey;
    mPortToTool.clear();
    char mainType[3];
    mainType[2] = '\0';
    char serialNumber[9];
    serialNumber[8] = '\0';
    char channel[2];

    for (unsigned int i = 0; i < portHandles.size(); i++) {
        CommandInitialize();
        CommandAppend("PHINF ");
        CommandAppend(portHandles[i].Pointer());
        CommandAppend("0021");  // 21 = 1 || 20
        CommandSend();
        ResponseRead();
        sscanf(mSerialBuffer, "%2c%*1c%*1c%*2c%*2c%*12c%*3c%8c%*2c%*8c%*2c%*2c%2c",
               mainType, serialNumber, channel);

        // create a unique pseudo-serialNumber to differentiate the second channel of Dual 5-DoF tools (Aurora only)
        if (strncmp(channel, "01", 2) == 0) {
            serialNumber[7] += 1;
        }

        /// \todo This is a workaround for an issue using the USB port on the latest Aurora
        if (strncmp(serialNumber, "00000000", 8) == 0) {
            CMN_LOG_CLASS_INIT_DEBUG << "PortHandlesQuery: received serial number of all zeros, skipping this tool and trying again" << std::endl;
            Sleep(0.5 * cmn_s);
            PortHandlesInitialize();
            PortHandlesQuery();
            return;
        }

        // generate a name and add (AddTool will skip existing tools)
        std::string name = std::string(mainType) + '-' + std::string(serialNumber);
        tool = AddTool(name, serialNumber, "");

        // update tool information
        sscanf(mSerialBuffer, "%2c%*1X%*1X%*2c%*2c%12c%3c%*8c%*2c%20c",
               tool->MainType, tool->ManufacturerID, tool->ToolRevision, tool->PartNumber);
        strncpy(tool->PortHandle, portHandles[i].Pointer(), 2);

        // associate the tool to its port handle
        toolKey = portHandles[i].Pointer();
        CMN_LOG_CLASS_INIT_VERBOSE << "PortHandlesQuery: associating " << tool->Name << " to port handle " << tool->PortHandle << std::endl;
        mPortToTool.AddItem(toolKey, tool, CMN_LOG_LEVEL_INIT_ERROR);

        CMN_LOG_CLASS_INIT_DEBUG << "PortHandlesQuery:\n"
                                 << " * Port Handle: " << tool->PortHandle << "\n"
                                 << " * Main Type: " << tool->MainType << "\n"
                                 << " * Manufacturer ID: " << tool->ManufacturerID << "\n"
                                 << " * Tool Revision: " << tool->ToolRevision << "\n"
                                 << " * Serial Number: " << tool->SerialNumber << "\n"
                                 << " * Part Number: " << tool->PartNumber << std::endl;
    }
}


void mtsNDISerial::PortHandlesEnable(void)
{
    char * parsePointer;
    unsigned int numPortHandles = 0;
    std::vector<vctChar3> portHandles;

    CommandSend("PHSR 03");
    ResponseRead();
    parsePointer = mSerialBuffer;
    sscanf(parsePointer, "%02X", &numPortHandles);
    parsePointer += 2;
    portHandles.resize(numPortHandles);
    for (unsigned int i = 0; i < portHandles.size(); i++) {
        sscanf(parsePointer, "%2c%*3c", portHandles[i].Pointer());
        parsePointer += 5;
        portHandles[i][2] = '\0';
    }
    for (unsigned int i = 0; i < portHandles.size(); i++) {
        CommandInitialize();
        CommandAppend("PENA ");
        CommandAppend(portHandles[i].Pointer());

        Tool * tool;
        std::string toolKey = portHandles[i].Pointer();
        tool = mPortToTool.GetItem(toolKey);
        if (!tool) {
            CMN_LOG_CLASS_RUN_ERROR << "PortHandlesEnable: no tool for port handle: " << toolKey << std::endl;
            return;
        }

        if (strncmp(tool->MainType, "01", 2) == 0) {  // reference
            CommandAppend("S");  // static
        } else if (strncmp(tool->MainType, "02", 2) == 0) {  // probe
            CommandAppend("D");  // dynamic
        } else if (strncmp(tool->MainType, "03", 2) == 0) {  // button box or foot switch
            CommandAppend("B");  // button box
        } else if (strncmp(tool->MainType, "04", 2) == 0) {  // software-defined
            CommandAppend("D");  // dynamic
        } else if (strncmp(tool->MainType, "0A", 2) == 0) {  // C-arm tracker
            CommandAppend("D");  // dynamic
        } else {
            CMN_LOG_CLASS_RUN_ERROR << "PortHandlesEnable: unknown tool of main type: " << tool->MainType << std::endl;
            return;
        }
        CommandSend();
        ResponseRead("OKAY");
        CMN_LOG_CLASS_RUN_DEBUG << "PortHandlesEnable: enabled port handle: " << portHandles[i].Pointer() << std::endl;
    }
}


void mtsNDISerial::PortHandlesPassiveTools(void)
{
    char portHandle[3];
    portHandle[2] = '\0';

    const ToolsType::const_iterator end = mTools.end();
    ToolsType::const_iterator toolIterator;
    for (toolIterator = mTools.begin();
         toolIterator != end;
         ++toolIterator) {
        if (toolIterator->second->Definition != "") {
            // request port handle for passive tool
            CommandSend("PHRQ *********1****");
            if (ResponseRead()) {
                sscanf(mSerialBuffer, "%2c", portHandle);
                CMN_LOG_CLASS_INIT_VERBOSE << "PortHandlesPassiveTools: loading "
                                           << toolIterator->first << " on port " << portHandle << std::endl;
                LoadToolDefinitionFile(portHandle, toolIterator->second->Definition);
                mPortToTool.AddItem(portHandle, toolIterator->second, CMN_LOG_LEVEL_INIT_ERROR);
            } else {
                CMN_LOG_CLASS_INIT_ERROR << "PortHandlesPassiveTools: failed to receive port handle for passive tool" << std::endl;
            }
        }
    }
}


void mtsNDISerial::ToggleTracking(const bool & track)
{
    // detect change
    if (track == mIsTracking) {
        return;
    }

    // if track requested
    if (track) {
        CommandSend("TSTART 80");
        if (ResponseRead("OKAY")) {
            mIsTracking = true;
            Events.Tracking(true);
            mControllerInterface->SendStatus(this->GetName() + ": tracking is on");
        } else {
            mControllerInterface->SendError(this->GetName() + ": failed to turn tracking on");
        }
    } else {
        CommandSend("TSTOP ");
        if (ResponseRead("OKAY")) {
            mIsTracking = false;
            Events.Tracking(false);
            mControllerInterface->SendStatus(this->GetName() + ": tracking is off");
        } else {
            mControllerInterface->SendError(this->GetName() + ": failed to turn tracking off");
        }
    }
    Sleep(0.5 * cmn_s);
}

void mtsNDISerial::ToggleStrayMarkers(const bool & stray)
{
    mTrackStrayMarkers = stray;
}

void mtsNDISerial::Track(void)
{
    char * parsePointer;
    unsigned int numPortHandles = 0;
    unsigned int numMarkers = 0;
    char portHandle[3];
    portHandle[2] = '\0';
    std::string toolKey;
    Tool * tool;
    vctQuatRot3 toolOrientation;
    vct3 toolPosition;
    vctFrm3 tooltipPosition;

    if (!mTrackStrayMarkers) {
        CommandSend("TX 0001");
    } else {
        CommandSend("TX 1001");
    }
    ResponseRead();
    parsePointer = mSerialBuffer;
    sscanf(parsePointer, "%02X", &numPortHandles);
    parsePointer += 2;
    CMN_LOG_CLASS_RUN_DEBUG << "Track: tracking " << numPortHandles << " tools" << std::endl;
    for (unsigned int i = 0; i < numPortHandles; i++) {
        sscanf(parsePointer, "%2c", portHandle);
        parsePointer += 2;
        toolKey = portHandle;
        tool = mPortToTool.GetItem(toolKey);
        if (!tool) {
            CMN_LOG_CLASS_RUN_ERROR << "Track: no tool for port handle: " << toolKey << std::endl;
            return;
        }

        if (strncmp(parsePointer, "MISSING", 7) == 0) {
            CMN_LOG_CLASS_RUN_VERBOSE << "Track: " << tool->Name << " is missing" << std::endl;
            tool->TooltipPosition.SetValid(false);
            tool->MarkerPosition.SetValid(false);
            parsePointer += 7;
            parsePointer += 8;  // skip Port Status
        } else if (strncmp(parsePointer, "DISABLED", 8) == 0) {
            CMN_LOG_CLASS_RUN_VERBOSE << "Track: " << tool->Name << " is disabled" << std::endl;
            tool->TooltipPosition.SetValid(false);
            tool->MarkerPosition.SetValid(false);
            parsePointer += 8;
            parsePointer += 8;  // skip Port Status
        } else if (strncmp(parsePointer, "UNOCCUPIED", 10) == 0) {
            CMN_LOG_CLASS_RUN_VERBOSE << "Track: " << tool->Name << " is unoccupied" << std::endl;
            tool->TooltipPosition.SetValid(false);
            tool->MarkerPosition.SetValid(false);
            parsePointer += 10;
            parsePointer += 8;  // skip Port Status
        } else {
            sscanf(parsePointer, "%6lf%6lf%6lf%6lf%7lf%7lf%7lf%6lf%*8X",
                   &(toolOrientation.W()), &(toolOrientation.X()), &(toolOrientation.Y()), &(toolOrientation.Z()),
                   &(toolPosition.X()), &(toolPosition.Y()), &(toolPosition.Z()),
                   &(tool->ErrorRMS));
            parsePointer += (4 * 6) + (3 * 7) + 6 + 8;

            toolOrientation.Divide(10000.0); // implicit format -x.xxxx
            tooltipPosition.Rotation().FromRaw(toolOrientation);
            toolPosition.Divide(100.0); // convert to mm, implicit format -xxxx.xx
            toolPosition.Multiply(cmn_mm); // convert to whatever cisst is using internally
            tooltipPosition.Translation() = toolPosition;
            tool->ErrorRMS /= 10000.0; // implicit format -x.xxxx
            tool->MarkerPosition.Position() = tooltipPosition; // Tool Frame Position = Orientation + Frame Origin
            tool->MarkerPosition.SetValid(true);

            tooltipPosition.Translation() += tooltipPosition.Rotation() * tool->TooltipOffset;  // apply tooltip offset
            tool->TooltipPosition.Position() = tooltipPosition;  // Tool Tip Position = Orientation + Tooltip
            tool->TooltipPosition.SetValid(true);
        }
        sscanf(parsePointer, "%08X", &(tool->FrameNumber));
        parsePointer += 8;
        CMN_LOG_CLASS_RUN_DEBUG << "Track: frame number: " << tool->FrameNumber << std::endl;
        if (*parsePointer != '\n') {
            CMN_LOG_CLASS_RUN_ERROR << "Track: line feed expected, received: " << *parsePointer << std::endl;
            return;
        }
        parsePointer += 1;  // skip line feed (LF)
    }

    if (mTrackStrayMarkers) {
        // read number of stray markers
        sscanf(parsePointer, "%02X", &numMarkers);
        parsePointer += 2;

        // read "out of volume" reply (see, API documentation for "Reply Option 1000" if this section seems convoluted)
        unsigned int outOfVolumeReplySize = static_cast<unsigned int>(ceil(numMarkers / 4.0));
        std::vector<bool> outOfVolumeReply(4 * outOfVolumeReplySize);
        unsigned int numGarbageBits = (4 * outOfVolumeReplySize) - numMarkers;
        for (unsigned int i = 0; i < outOfVolumeReplySize; i++) {
            std::bitset<4> outOfVolumeReplyByte(parsePointer[i]);
            outOfVolumeReplyByte.flip();
            for (unsigned int j = 0; j < 4; j++) {
                outOfVolumeReply[4*i + j] = outOfVolumeReplyByte[3-j];  // 0 if out of volume
            }
        }
        parsePointer += outOfVolumeReplySize;

        // read marker positions
        std::vector<vct3> markerPositions(numMarkers);
        std::vector<bool> markerVisibilities(numMarkers);
        mStrayMarkers.Zeros();
        for (unsigned int i = 0; i < numMarkers; i++) {
            sscanf(parsePointer, "%7lf%7lf%7lf",
                   &(markerPositions[i].X()), &(markerPositions[i].Y()), &(markerPositions[i].Z()));
            parsePointer += (3 * 7);
            markerPositions[i].Divide(100.0);  // handle the implied decimal point
            markerVisibilities[i] = outOfVolumeReply[i + numGarbageBits];  // handle garbage bits in reply

            mStrayMarkers[i][0] = 1.0;  // if a marker is encountered
            mStrayMarkers[i][1] = markerVisibilities[i];  // if marker is NOT out of volume
            mStrayMarkers[i][2] = markerPositions[i].X();
            mStrayMarkers[i][3] = markerPositions[i].Y();
            mStrayMarkers[i][4] = markerPositions[i].Z();

            std::cerr << "ReportStrayMarkers: " << i + 1
                      << "th marker visibility: " << markerVisibilities[i]
                      << ", position: " << markerPositions[i] << std::endl;
        }
    }
    parsePointer += 4;  // skip System Status
}


void mtsNDISerial::ReportStrayMarkers(void)
{
    char * parsePointer;
    unsigned int numPortHandles = 0;
    unsigned int numMarkers = 0;

    // save tracking status
    bool wasTracking = mIsTracking;
    ToggleTracking(true);

    CommandSend("TX 1000");
    ResponseRead();
    parsePointer = mSerialBuffer;

    // skip handle number for all port handles
    sscanf(parsePointer, "%02X", &numPortHandles);
    parsePointer += 2;
    for (unsigned int i = 0; i < numPortHandles; i++) {
        parsePointer += 2;  // skip handle number
        parsePointer += 1;  // skip line feed (LF)
    }

    // read number of stray markers
    sscanf(parsePointer, "%02X", &numMarkers);
    parsePointer += 2;
    CMN_LOG_CLASS_RUN_DEBUG << "ReportStrayMarkers: " << numMarkers << " stray markers detected" << std::endl;

    // read "out of volume" reply (see, API documentation for "Reply Option 1000" if this section seems convoluted)
    unsigned int outOfVolumeReplySize = static_cast<unsigned int>(ceil(numMarkers / 4.0));
    std::vector<bool> outOfVolumeReply(4 * outOfVolumeReplySize);
    unsigned int numGarbageBits = (4 * outOfVolumeReplySize) - numMarkers;
    for (unsigned int i = 0; i < outOfVolumeReplySize; i++) {
        std::bitset<4> outOfVolumeReplyByte(parsePointer[i]);
        outOfVolumeReplyByte.flip();
        for (unsigned int j = 0; j < 4; j++) {
            outOfVolumeReply[4*i + j] = outOfVolumeReplyByte[3-j];  // 0 if out of volume
        }
    }
    parsePointer += outOfVolumeReplySize;

    // read marker positions
    std::vector<vct3> markerPositions(numMarkers);
    std::vector<bool> markerVisibilities(numMarkers);
    mStrayMarkers.Zeros();
    for (unsigned int i = 0; i < numMarkers; i++) {
        sscanf(parsePointer, "%7lf%7lf%7lf",
               &(markerPositions[i].X()), &(markerPositions[i].Y()), &(markerPositions[i].Z()));
        parsePointer += (3 * 7);
        markerPositions[i].Divide(100.0);  // handle the implied decimal point
        markerVisibilities[i] = outOfVolumeReply[i + numGarbageBits];  // handle garbage bits in reply

        mStrayMarkers[i][0] = 1.0;  // if a marker is encountered
        mStrayMarkers[i][1] = markerVisibilities[i];  // if marker is NOT out of volume
        mStrayMarkers[i][2] = markerPositions[i].X();
        mStrayMarkers[i][3] = markerPositions[i].Y();
        mStrayMarkers[i][4] = markerPositions[i].Z();

        CMN_LOG_CLASS_RUN_DEBUG << "ReportStrayMarkers: " << i + 1
                                << "th marker visibility: " << markerVisibilities[i]
                                << ", position: " << markerPositions[i] << std::endl;
    }
    parsePointer += 4;  // skip System Status

    // restore tracking status
    ToggleTracking(wasTracking);
}


mtsNDISerial::Tool::Tool(void) :
    TooltipOffset(0.0)
{
    PortHandle[2] = '\0';
    MainType[2] = '\0';
    ManufacturerID[12] = '\0';
    ToolRevision[3] = '\0';
    SerialNumber[8] = '\0';
    PartNumber[20] = '\0';
}
