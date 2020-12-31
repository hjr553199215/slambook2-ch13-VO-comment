#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

/*<>和""表示编译器在搜索头文件时的顺序不同，<>表示从系统目录下开始搜索，然后再搜索PATH环境变量所列出的目录，
 * 不搜索当前目录，""是表示从当前目录开始搜索，然后是系统目录和PATH环境变量所列出的目录。
    所以，系统头文件一般用<>，用户自己定义的则可以使用""，加快搜索速度。
    使用<>这种方式，编译器查找的时候，会在编译器的安装目录的标准库中开始查找，
    ""这种方式，会在当前的工程所在的文件夹开始寻找，也就是你的源程序所在的文件夹。有的编译器，要求十分严格，不能混用，有的就可以。*/


#include "myslam/common_include.h"


namespace myslam{

    // forward declare
    struct MapPoint;  //这里为什么要申明MapPoint
    struct Feature;

/**
 * 帧
 * 每一帧分配独立id，关键帧分配关键帧ID
 */

    struct Frame {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr; //无论Frame还是Feature还是MapPoint，后面都是在用智能指针类型在构建对象，所以这里有这句智能指针定义

        unsigned long id_=0; //id of current frame ,,可以在struct内直接对成员变量直接赋值吗？
        unsigned long keyframe_id_ = 0; //id of keyframe
        bool is_keyframe_=false;
        double time_stamp_;
        SE3 pose_ ;//T李群形式的变换矩阵
        std::mutex pose_mutex_;          // Pose数据锁
        cv::Mat left_img_, right_img_;   // 双目图像，左目，右目

        std::vector<std::shared_ptr<Feature>> features_left_;
        std::vector<std::shared_ptr<Feature>> features_right_;
/*shared_ptr是一种智能指针（smart pointer），作用有如同指针，但会记录有多少个shared_ptrs共同指向一个对象。这便是所谓的引用计数
 * （reference counting）,比如我们把只能指针赋值给另外一个对象,那么对象多了一个智能指针指向它,所以这个时候引用计数会增加一个,
 * 我们可以用shared_ptr.use_count()函数查看这个智能指针的引用计数,一旦最后一个这样的指针被销毁，
 * 也就是一旦某个对象的引用计数变为0，这个对象会被自动删除,当我们程序结束进行return的时候,智能指针的引用计数会减1,
*/

//构造函数

        Frame() {}

        Frame(long id, double time_stamp, const SE3 &pose, const Mat &left,
              const Mat &right);
        //函数参数中*为指针传递参数，&为引用传递参数，前面没有标识符时为值传递参数
        /*    函数里面的是形式参数，形参;    实际传入函数的变量是实际参数，实参
         * 值传递：
               形参是实参的拷贝，改变形参的值并不会影响外部实参的值。从被调用函数的角度来说，值传递是单向的（实参->形参），参数的值只能传入，
               不能传出。当函数内部需要修改参数，并且不希望这个改变影响调用者时，采用值传递。
           指针传递：
               形参为指向实参地址的指针，当对形参的指向操作时，就相当于对实参本身进行的操作
           引用传递：
               形参相当于是实参的“别名”，对形参的操作其实就是对实参的操作，在引用传递过程中，被调函数的形式参数虽然也作为局部变量在栈
               中开辟了内存空间，但是这时存放的是由主调函数放进来的实参变量的地址。被调函数对形参的任何操作都被处理成间接寻址，即通过
               栈中存放的地址访问主调函数中的实参变量。正因为如此，被调函数对形参做的任何操作都影响了主调函数中的实参变量。
         */


        SE3 Pose() {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            return pose_;
        }

        void SetPose(const SE3 &pose) {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        /// 设置关键帧并分配并键帧id
        void SetKeyFrame();

        /// 工厂构建模式，分配id
        static std::shared_ptr<Frame> CreateFrame();

    };
}

#endif  // MYSLAM_FRAME_H

