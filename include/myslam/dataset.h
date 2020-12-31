#ifndef MYSLAM_DATASET_H
#define MYSLAM_DATASET_H
#include "myslam/camera.h"
#include "myslam/common_include.h"
#include "myslam/frame.h"


///数据集操作类

namespace myslam{


    class Dataset{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//内存位数对齐，反正加上就对了吧

        typedef std::shared_ptr<Dataset> Ptr;

        Dataset(const std::string& dataset_path);//构造函数

        /// 初始化，返回是否成功
        bool Init();//初始化函数，为什么这里需要初始化呢？

        /// create and return the next frame containing the stereo images
        Frame::Ptr NextFrame();

        Camera::Ptr GetCamera(int camera_id) const {
            return cameras_.at(camera_id);
        }


    private:

        std::string dataset_path_;
        int current_image_index_ = 0;

        std::vector<Camera::Ptr> cameras_;


    };

}

#endif