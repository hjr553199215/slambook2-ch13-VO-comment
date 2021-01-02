#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <thread>
#include <pangolin/pangolin.h>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"


//接下来进入可视化环节，通过Pangolin将视觉里程计中的帧,点和行进路径可视化


namespace myslam{

    class Viewer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Viewer> Ptr;

        Viewer();//构造函数

        void SetMap(Map::Ptr map) { map_ = map; } //地图类成员变量map_是个私有变量，需要通过函数接口SetMap来赋值

        void Close();

        // 增加一个当前帧
        void AddCurrentFrame(Frame::Ptr current_frame);

        // 更新地图
        void UpdateMap();


    private:


        void ThreadLoop();

        void DrawFrame(Frame::Ptr frame, const float* color);  //画出关键帧

        void DrawMapPoints();//画出地图点

        void FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera);//视角跟随当前帧


        cv::Mat PlotFrameImage();//将当前帧中的特征画在图像上

        Frame::Ptr current_frame_ = nullptr;
        Map::Ptr map_ = nullptr;

        std::thread viewer_thread_;
        bool viewer_running_ = true;

        std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
        std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;
        bool map_updated_ = false;

        std::mutex viewer_data_mutex_;
    };

}

#endif