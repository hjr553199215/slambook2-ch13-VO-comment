#include "myslam/visual_odometry.h"
#include <chrono>
#include "myslam/config.h"

namespace myslam{

    VisualOdometry::VisualOdometry(std::string &config_path)
            : config_file_path_(config_path) {}

    bool VisualOdometry::Init()
    {
        if (Config::SetParameterFile(config_file_path_) == false) {
            //判断一下这个config_file_path_是否存在，同时将配置文件赋给Config类中的cv::FileStorage file_,便于对文件操作
            return false;
        }

        dataset_ = Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));//数据集类初始化，读相机参数
        //1.模板函数自动类型推导调用Config::Get("dataset_dir")；
        //2.模板函数具体类型显示调用Config::Get<std::string>("dataset_dir")；

        CHECK_EQ(dataset_->Init(), true);//功能类似assert断言，但不受DEBUG模式控制即非DEBUG模式也生效


        //接下来按照逻辑关系一层层的确立联系，一个完整的VO包含前端,后端,地图,可视化器等模块，因此有下述创建代码
        frontend_ = Frontend::Ptr(new Frontend);
        backend_ = Backend::Ptr(new Backend);
        map_ = Map::Ptr(new Map);
        viewer_ = Viewer::Ptr(new Viewer);

        //在一个VO中，前端需对应后端,地图,可视化器,相机类等,这在frontend的类定义中有清楚显示，所以将它们连接起来
        frontend_->SetBackend(backend_);
        frontend_->SetMap(map_);
        frontend_->SetViewer(viewer_);
        frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));


        //后端类的定义中用到了相机类和地图类，所以要将后端类与相机类和地图类连接起来
        backend_->SetMap(map_);
        backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));


        //对于可视化器来说，只要有地图就可以，它只是将地图可视化，所以不需要其它模块，只需将其与地图模块连接在一起
        viewer_->SetMap(map_);

        return true;
    };


    void VisualOdometry::Run()
    {
        while (1) {
            LOG(INFO) << "VO is running";
            if (Step() == false) {
            //这里的主过程执行在这条if语句中,每次做条件判断都需要执行Step()，即步进操作，如果步进出问题，则跳出死循环while (1)
                break;
            }
        }

        //如果运行出错，则停止后端，关闭可视化器，同时因为跳出死循环，所以不再执行step(),所以前端也不再对图像序列进行步进跟踪
        backend_->Stop();
        viewer_->Close();

        LOG(INFO) << "VO exit";
    };


    bool VisualOdometry::Step()
    {
        Frame::Ptr new_frame = dataset_->NextFrame();//从数据集中读出下一帧
        if (new_frame == nullptr) return false;//如果读到的下一帧为空，也就是说没有读到下一帧，则无法继续跟踪，报错

        auto t1 = std::chrono::steady_clock::now();
        bool success = frontend_->AddFrame(new_frame);//将新的一帧加入到前端中，进行跟踪处理,帧间位姿估计
        auto t2 = std::chrono::steady_clock::now();
        auto time_used =
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";

        return true;
    }
}