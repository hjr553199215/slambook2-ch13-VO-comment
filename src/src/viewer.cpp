#include "myslam/viewer.h"
#include "myslam/feature.h"
#include "myslam/frame.h"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>


namespace myslam{

    Viewer::Viewer()
    {
        viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
    }

    void Viewer::Close() {
        viewer_running_ = false;
        viewer_thread_.join();
    }

    void Viewer::AddCurrentFrame(Frame::Ptr current_frame) {
        std::unique_lock<std::mutex> lck(viewer_data_mutex_);
        current_frame_ = current_frame;
    }

    cv::Mat Viewer::PlotFrameImage() {
        cv::Mat img_out;
        cv::cvtColor(current_frame_->left_img_, img_out, CV_GRAY2BGR);
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            if (current_frame_->features_left_[i]->map_point_.lock()) {
                auto feat = current_frame_->features_left_[i];
                cv::circle(img_out, feat->position_.pt, 2, cv::Scalar(0, 250, 0),
                           2);
            }
        }
        return img_out;
    }


    void Viewer::UpdateMap() {
        std::unique_lock<std::mutex> lck(viewer_data_mutex_);
        assert(map_ != nullptr);//assert() 的用法像是一种"判断式编程"，在我的理解中，其表达的意思就是，
        // 程序在我的假设条件下，能够正常良好的运作，其实就相当于一个 if 语句：
        active_keyframes_ = map_->GetActiveKeyFrames();
        active_landmarks_ = map_->GetActiveMapPoints();
        map_updated_ = true;
    }


    void Viewer::ThreadLoop() {
        pangolin::CreateWindowAndBind("MySLAM", 1024, 768); //创建一个GUI界面

        //glEnable用法可参照：https://www.cnblogs.com/icmzn/articles/5741484.html
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);

        //glEnable和glBlendFunc可以参照：https://blog.csdn.net/ZhaDeNianQu/article/details/103926074
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState vis_camera(   //向界面中放置一个相机
                pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
                pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));
        //这里第一组参数ex,ey,ez的意思是相机在哪里，第二组参数lx,ly,lz的意思是看向哪个点，第三组参数的意思是Pangolin坐标系哪个轴指向屏幕上方
        ///Pangolin中空间几何关系还是比较抽象的，如果到这里看不明白的话也不要紧，可以先跳过，后续对Pangolin接触多后自然就可以理解
        ///OPENGL参考： https://learnopengl-cn.readthedocs.io/zh/latest/01%20Getting%20started/09%20Camera/

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& vis_display =
                pangolin::CreateDisplay()
                        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                        .SetHandler(new pangolin::Handler3D(vis_camera));
        //在先前创建的GUI界面中添加创建一个视口，或者说是一个可视化窗口，可以理解为从GUI界面中划分出特定大小的一部分窗口用于显示特定内容

        const float blue[3] = {0, 0, 1};
        const float green[3] = {0, 1, 0};

        while (!pangolin::ShouldQuit() && viewer_running_) //循环刷新，两个终止条件，第一个就是叫停Pangolin，第二个就是VO失败，可视化停止。
        {

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//清空颜色缓存和深度缓存，即清除屏幕
            //glClear和glClearColor的含义和用法可参照:http://blog.sina.com.cn/s/blog_15939d9370102xasd.html
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            vis_display.Activate(vis_camera);//将视口vis_display根据vis_camera这个相机视角激活为一个交互式视图（可以用鼠标进行视角旋转）

            std::unique_lock<std::mutex> lock(viewer_data_mutex_);
            if (current_frame_) { //如果当前帧不为空指针
                DrawFrame(current_frame_, green);
                FollowCurrentFrame(vis_camera);  //让可视化窗口跟着当前帧走，即跟随状态

                cv::Mat img = PlotFrameImage();//把特征点在图像上画出来
                cv::imshow("image", img);
                cv::waitKey(1);//停1ms
            }

            if (map_) {
                //如果map_不为空则执行后续,map_这个类成员变量通过.h文件中定义的SetMap()函数来赋值，
                // 在其它线程中完成地图更新后可以用SetMap()函数为接口传入可视化器Viewer类对象.
                DrawMapPoints();  //把点云画出来
            }

            pangolin::FinishFrame();
            usleep(5000);//usleep功能把进程挂起一段时间， 单位是微秒（百万分之一秒）；
        }

        LOG(INFO) << "Stop viewer";
    }

    void Viewer::DrawFrame(Frame::Ptr frame, const float* color) {
        SE3 Twc = frame->Pose().inverse();
        const float sz = 1.0;
        const int line_width = 2.0;
        const float fx = 400;
        const float fy = 400;
        const float cx = 512;
        const float cy = 384;
        const float width = 1080;
        const float height = 768;


        //关于glPushMatrix()和glPopMatrix()操作网上有一个很有趣的解释，希望能帮助你理解：
        //有时候在经过一些变换后我们想回到原来的状态，就像我们谈恋爱一样，换来换去还是感觉初恋好
        //参考资料：https://blog.csdn.net/passtome/article/details/7768379
        glPushMatrix();//压栈操作

        Sophus::Matrix4f m = Twc.matrix().template cast<float>();
        glMultMatrixf((GLfloat*)m.data());//将原有的世界坐标系变换到当前坐标系下，便于直接画出相机模型

        if (color == nullptr) {   //这里决定了Pangolin中画出来的相机颜色
            glColor3f(1, 0, 0);
        } else
            glColor3f(color[0], color[1], color[2]);

        //这里要在Pangolin里面画出相机模型了，如果你实际运行了这个VO，应该会记得Pangolin里面画出的那个小相机是由8条边构成的
        glLineWidth(line_width);//设置线宽
        glBegin(GL_LINES);//开始画线
        //一个起点，一个终点就确定了一条构成相机模型的线，一个模型8条边，因此下面一共有16各顶点定义。
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glEnd();  //结束画线
        glPopMatrix();  //弹出操作，或者说出栈操作，恢复到这个变换以前的状态
    }

    void Viewer::DrawMapPoints() {
        const float red[3] = {1.0, 0, 0};
        for (auto& kf : active_keyframes_) {//将所有的窗口内激活帧都画成红色
            DrawFrame(kf.second, red);
        }

        glPointSize(2);//确定点云尺寸
        glBegin(GL_POINTS);//开始画点
        for (auto& landmark : active_landmarks_) {
            auto pos = landmark.second->Pos();  //获得地图路标点的世界坐标，存储在pos中
            glColor3f(red[0], red[1], red[2]);  //确定点云颜色为红
            glVertex3d(pos[0], pos[1], pos[2]);//给出点云位置，画出点云
        }
        glEnd();//结束画点云
    }

    void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
        SE3 Twc = current_frame_->Pose().inverse();
        pangolin::OpenGlMatrix m(Twc.matrix()); //依据Pangolin形式设定变换矩阵
        vis_camera.Follow(m, true);//交互视图的视角，或者说Pangolin观察相机的相机视角跟着这个变换矩阵进行随动，完成对当前帧的视角跟随
    }


}