///针对frame.h中涉及的各类函数进行定义实现


#include "myslam/frame.h"

namespace myslam {

    Frame::Frame(long id, double time_stamp, const SE3 &pose, const Mat &left, const Mat &right)
            : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right) {}

    Frame::Ptr Frame::CreateFrame() {
        static long factory_id = 0;   ///这里是不是有些问题？
        Frame::Ptr new_frame(new Frame);
        new_frame->id_ = factory_id++;
        return new_frame;
    }

    void Frame::SetKeyFrame() {
        static long keyframe_factory_id = 0;
        is_keyframe_ = true;
        keyframe_id_ = keyframe_factory_id++;
    }

}