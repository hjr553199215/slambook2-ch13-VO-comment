#include "myslam/config.h"

namespace myslam{

    bool Config::SetParameterFile(const std::string &filename) {
        if (config_ == nullptr)
            config_ = std::shared_ptr<Config>(new Config);
        config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
        if (config_->file_.isOpened() == false) {
            LOG(ERROR) << "parameter file " << filename << " does not exist.";
            config_->file_.release();
            return false;
        }
        return true;
    }

    Config::~Config() {
        if (file_.isOpened())
            file_.release();
    }

    std::shared_ptr<Config> Config::config_ = nullptr;
//一般来说无论怎样静态成员变量都需要在类外进行定义，静态成员变量在类外定义且初始化,
// 这里初始化的时候之所以只声明std::shared_ptr<Config> 类型，不带static关键字
//是因为关键字static只出现在类的内部,在外部对静态成员变量初始化的时候不加


}

