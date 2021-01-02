


///在include和src文件夹中，我们已经定义并实现了各种类及函数，现在我们只需要在本文件中为整个视觉里程计VO提供一个入口即可

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

//gflags库的使用说明参考： https://www.jianshu.com/p/2179938a818d
//同上：https://blog.csdn.net/NMG_CJS/article/details/104436079?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.control

DEFINE_string(config_file, "/home/hujiarui/slambook2-master/ch13/config/default.yaml", "config file path");//第三个参数是说明信息

int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);

    myslam::VisualOdometry::Ptr vo(
            new myslam::VisualOdometry(FLAGS_config_file));
    //构造一个VO类对象，这里其实FLAGS_config_file也可以用argv[]里面的内容来替代。
    assert(vo->Init() == true);//这里虽然是一个判断语句，但是在该过程中已然完成了VO类对象vo的初始化
    vo->Run();//把视觉里程计跑起来吧！

    return 0;
}