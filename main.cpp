#include "logging.h"
#include "common.h"
#include "iostream"
#include "encoderdecoder_infer.h"
#include <cmath>
#include <iomanip>
#include "cmdline.h"
#include "httplib.h"
#include <nlohmann/json.hpp>
#include "spdlog/spdlog.h"


// std::shared_ptr<spdlog::logger> g_Logger = nullptr;
// #define SPDLOG_INFO(format, ...) SPDLOG_LOGGER_INFO(g_Logger, format, __VA_ARGS__)
// #define SPDLOG_ERROR(format, ...) SPDLOG_LOGGER_ERROR(g_Logger, format, __VA_ARGS__)

std::shared_ptr<T5Inference>  encoderdecoder_handle = nullptr;

#ifdef _WIN32
unsigned long get_tick_count()
{
    unsigned long tick = GetTickCount64();

    return tick;
}
int GetIPAddr(std::string &ip_addr)
{
    char szText[256];
    // 获取本机主机名称
    int iRet;
    iRet = gethostname(szText, 256);
    int a = WSAGetLastError();
    if (iRet != 0)
    {
        printf("gethostname() Failed!");
        return false;
    }
    // 通过主机名获取到地址信息
    HOSTENT *host = gethostbyname(szText);
    if (NULL == host)
    {
        printf("gethostbyname() Failed!");
        return false;
    }
    in_addr PcAddr;
    for (int i = 0;; i++)
    {
        char *p = host->h_addr_list[i];
        if (NULL == p)
        {
            break;
        }
        memcpy(&(PcAddr.S_un.S_addr), p, host->h_length);
        char *szIP = ::inet_ntoa(PcAddr);
        ip_addr = szIP;
        printf("本机的ip地址是：%s\n", szIP);
        return true;
    }
    return false;
}
#else
static std::vector<std::string> splitByDelimiter2(const std::string& s, char delimiter) {
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens;
}
int GetIPAddr(std::string &ip_addr)
{
	FILE *fp = popen("hostname -I", "r");
    if (fp == nullptr) {
        std::cerr << "Failed to run command hostname" << std::endl;
        return 0;
    }
	char buffer[1024];
    while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
		std::string ip_str=buffer;		
		std::vector<std::string>str_vec=splitByDelimiter2(ip_str,' ');
		if(str_vec.size()>0)
		{
			ip_addr=str_vec.at(0);
			printf("本机的ip地址是：%s\n", ip_addr.c_str());
			return 1;
		}		
    }
    pclose(fp);
    return 0;
    pclose(fp);
}
// milliseconds
unsigned long get_tick_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    unsigned long tick = static_cast<unsigned long>(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
    return tick;
}
#endif

int trigger_T5(const httplib::Request &req, httplib::Response &res)
{
    try
    {
        SPDLOG_INFO("[T5]收到请求 Request req={}.", req.body);
        nlohmann::json req_json;
        nlohmann::json res_json;
        try
        {
            req_json = nlohmann::json::parse(req.body);
            if (req_json["InputText"].is_null())
            {
                SPDLOG_ERROR("[T5]The req is not vaild. input_text is null{}.", req.body);
                res_json["StatusCode"] = 400;
                res_json["Message"] = "RequestID error";
                res_json["RequestID"] = "RequestID";
                SPDLOG_ERROR("res=.{}", res_json.dump(true));
                res.set_content(res_json.dump(true), "application/json");

                return false;
            }
            if (req_json["RequestID"].is_null())
            {
                SPDLOG_ERROR("[T5]The req is not vaild. RequestID is null{}.", req.body);

                res_json["StatusCode"] = 400;
                res_json["Message"] = "RequestID error";
                res_json["RequestID"] = "RequestID";
                SPDLOG_ERROR("res=.{}", res_json.dump(true));
                res.set_content(res_json.dump(true), "application/json");

                return false;
            }

            std::string InputText = req_json["InputText"];   
            std::string RequestID = req_json["RequestID"];
            try
            {
                std::string T5Result =encoderdecoder_handle->InferEncoderDecoder(InputText);
                if (T5Result.size() != 0)
                {                 
                    res_json["InputText"] = InputText;
                    res_json["T5Result"] = T5Result;
                    SPDLOG_INFO("[T5]T5Result={}", T5Result);
                    res.status = 200;
                    res_json["StatusCode"] = 200;
                    res_json["Message"] = "succeed";
                    res_json["RequestID"] = RequestID;

                    SPDLOG_INFO("[T5]res=.{}", res_json.dump(true));
                    res.set_content(res_json.dump(true), "application/json");
                    SPDLOG_INFO("[T5]trigger_T5响应成功");
                }
            }
            catch (std::exception e)
            {
                SPDLOG_ERROR("[T5] T5 infer error.{}", e.what());
                res_json["StatusCode"] = 400;
                res_json["Message"] = "T5 infer error";
                res_json["RequestID"] = RequestID;
                SPDLOG_ERROR("res=.{}", res_json.dump(true));
                res.set_content(res_json.dump(true), "application/json");
                return false;
            }
        }
        catch (std::exception e)
        {
            SPDLOG_ERROR("[T5]The input is not vaild.{}", req.body);
            res_json["StatusCode"] = 400;
            res_json["Message"] = "req error";
            res_json["RequestID"] = "RequestID";
            SPDLOG_ERROR("res=.{}", res_json.dump(true));
            res.set_content(res_json.dump(true), "application/json");

            return false;
        }
    }
    catch (const std::exception &ex)
    {
        nlohmann::json res_json;
        SPDLOG_ERROR("[T5]trigger_T5 Caught exception:{}", ex.what());
        res_json["StatusCode"] = 400;
        res_json["Message"] = "req error";
        res_json["RequestID"] = "RequestID";
        SPDLOG_ERROR("res=.{}", res_json.dump(true));
        res.set_content(res_json.dump(true), "application/json");
        return 0;
    }
}
int main(int argc, char** argv)
{
    cmdline::parser t5Args;
    t5Args.add<std::string>("input_text", 'i', "input_text", false, "translate English to French: I was a victim of a series of accidents.");
    t5Args.add<std::string>("enc_model_path", 'e', "enc_model_path", false, "/home/wzp/t5-onnx/t5-base-encoder-12.engine");
    t5Args.add<std::string>("dec_model_path", 'd', "dec_model_path", false, "/home/wzp/t5-onnx/t5-base-decoder-with-lm-head-12.engine");
    
    t5Args.add<std::string>("spiece_model_path", 's', "spiece_model_path", false, "/home/wzp/t5-onnx/spiece.model");
    t5Args.add<std::string>("ip_addr", 'r', "ip_addr", false, "");
    t5Args.add<int>("ip_port", 'p', "ip_port", false, 17653);
    t5Args.add<int>("use_mode", 'u', "use_mode ,0:demo,1:test,2:api", false, 0);
    try {
		t5Args.parse_check(argc, argv);
	}
	catch (const std::runtime_error& e) {
		SPDLOG_ERROR("参数错误:{}", e.what()) ;
		return 1;
	}
    std::string input_text = t5Args.get<std::string>("input_text");
    std::string enginePath = t5Args.get<std::string>("enc_model_path");
    std::string enginePathDec = t5Args.get<std::string>("dec_model_path"); 
    std::string spiece_model_path = t5Args.get<std::string>("spiece_model_path");
    std::string ip_addr = t5Args.get<std::string>("ip_addr");
    int ip_port = t5Args.get<int>("ip_port");
    int use_mode = t5Args.get<int>("use_mode");
    SPDLOG_INFO("input_text:{}", input_text);
    SPDLOG_INFO("enginePath:{}", enginePath);
    SPDLOG_INFO("enginePathDec:{}", enginePathDec);
    SPDLOG_INFO("ip_addr:{}", ip_addr);
    SPDLOG_INFO("use_mode:{}", use_mode);
    const int maxBatchSize=1;
    const int seqLength=100;
    const bool enableGraph = true;
    encoderdecoder_handle = std::make_shared<T5Inference>();
    int  ret=  encoderdecoder_handle->Init(enginePath,enginePathDec,spiece_model_path, maxBatchSize,seqLength);
    gLogInfo << "encoder_model done!"<<enginePath <<"\n";   
    if (use_mode == 0)
    {
        std::string result_text=encoderdecoder_handle->InferEncoderDecoder(input_text);
        gLogInfo << "result_text="<<result_text<<"\n";   
    }
    else if (use_mode == 1)
    {
        int test_num = 20;
        long time_start = get_tick_count();
        for (int i = 0; i < test_num; i++)
        {
            gLogInfo << i + 1 << "/" << test_num << "\n";
            std::string result = encoderdecoder_handle->InferEncoderDecoder(input_text);
        }
        long total_time = get_tick_count() - time_start;
        // // 计算耗时
        gLogInfo << "avg time={}ms" << (int)(total_time / float(test_num)) << "\n";
        // encoderdecoder_handle->reportTiming(1,1,test_num);
    }
   else if (use_mode == 2) // API
    {
        if (ip_addr == "")
        {
            if (!GetIPAddr(ip_addr))
            {
                SPDLOG_ERROR("获取本机IP失败");
                return 0;
            }
        }

        // HTTP
        httplib::Server svr;
        svr.Post("/T5/register", trigger_T5);
        SPDLOG_INFO("[T5]listenInThread = [http://{}:{}]", ip_addr, ip_port);
        svr.listen(ip_addr, ip_port);
    }

}
