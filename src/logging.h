#ifndef INFER_C_LOGGING_H
#define INFER_C_LOGGING_H

#include <NvInfer.h>
#include <cassert>
#include <iostream>
#include <map>

using namespace nvinfer1;
using Severity = nvinfer1::ILogger::Severity;
static std::string  getCurFileName(const std::string filepath)
{
    // std::string filepath = __FILE__;
    size_t lastSlashPos = filepath.find_last_of("\\/");
    std::string fileName = filepath.substr(lastSlashPos + 1);
    return fileName;
}

class Logger : public ILogger
{
public:
    Logger(Severity severity)
        : mOstream(&std::cout)
        , mReportableSeverity(severity)
    {
    }

    template <typename T>
    Logger& operator<<(T const& obj)
    {
        if (mOstream != nullptr)
        {
            *mOstream << obj;
        }
        return *this;
    }

    Logger& report(Severity severity, const char* file, int line, const char* msg)
    {
        if (severity <= mReportableSeverity)
        {
            const std::map<Severity, std::string> prefixMapping = {{Severity::kINTERNAL_ERROR, "[DemoT5][F] "},
                {Severity::kERROR, "[DemoT5][E] "}, {Severity::kWARNING, "[DemoT5][W] "},
                {Severity::kINFO, "[DemoT5][I] "}, {Severity::kVERBOSE, "[DemoT5][V] "}};

            assert(prefixMapping.find(severity) != prefixMapping.end());

            mOstream = &std::cout;

            *this << prefixMapping.at(severity) << "(" << getCurFileName(file) << ":" << line << ") " << msg;

            return *this;
        }
        mOstream = nullptr;
        return *this;
    }

private:
    void log(Severity severity, const char* msg) noexcept override
    {
        report(severity, __FILE__, __LINE__, msg) << "\n";
    }

    std::ostream* mOstream;
    Severity mReportableSeverity;
};

extern Logger gLogger;
#define gLogFatal gLogger.report(Severity::kINTERNAL_ERROR, __FILE__, __LINE__, "")
#define gLogError gLogger.report(Severity::kERROR, __FILE__, __LINE__, "")
#define gLogWarning gLogger.report(Severity::kWARNING, __FILE__, __LINE__, "")
#define gLogInfo gLogger.report(Severity::kINFO, __FILE__, __LINE__, "")
#define gLogVerbose gLogger.report(Severity::kVERBOSE, __FILE__, __LINE__, "")

#endif // INFER_C_LOGGING_H
