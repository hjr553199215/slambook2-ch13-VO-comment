///对map.h中声明的各类函数进行定义实现


#include "myslam/map.h"
#include "myslam/feature.h"

namespace myslam{

    void Map::CleanMap() {//RemoveOldKeyframe操作过后，有些帧被舍弃了，里面的特征观测也全部被丢弃，这样就可能造成有些landmark没有观测了
        int cnt_landmark_removed = 0;
        for (auto iter = active_landmarks_.begin();
             iter != active_landmarks_.end();) {

            if (iter->second->observed_times_ == 0) {//这里的意思是哪个landmark的观测次数等于0,
                //则将这个landmark从active_landmarks_中移除
                iter = active_landmarks_.erase(iter);
                cnt_landmark_removed++;
            } else {
                ++iter;
            }
        }
        LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
    }

    void Map::RemoveOldKeyframe() {
        if (current_frame_ == nullptr) return;
        // 寻找与当前帧最近与最远的两个关键帧
        double max_dis = 0, min_dis = 9999;
        double max_kf_id = 0, min_kf_id = 0;
        auto Twc = current_frame_->Pose().inverse();
        for (auto& kf : active_keyframes_) {
            if (kf.second == current_frame_) continue;
            auto dis = (kf.second->Pose() * Twc).log().norm();
            if (dis > max_dis) {
                max_dis = dis;
                max_kf_id = kf.first;
            }
            if (dis < min_dis) {
                min_dis = dis;
                min_kf_id = kf.first;
            }
        }

        const double min_dis_th = 0.2;  // 最近阈值
        Frame::Ptr frame_to_remove = nullptr;
        if (min_dis < min_dis_th) {
            // 如果存在很近的帧，优先删掉最近的
            frame_to_remove = keyframes_.at(min_kf_id);
        } else {
            // 删掉最远的
            frame_to_remove = keyframes_.at(max_kf_id);
        }

        LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;
        // remove keyframe and landmark observation
        active_keyframes_.erase(frame_to_remove->keyframe_id_);
        for (auto feat : frame_to_remove->features_left_) {
            auto mp = feat->map_point_.lock();
            if (mp) {
                mp->RemoveObservation(feat);
            }
        }
        for (auto feat : frame_to_remove->features_right_) {
            if (feat == nullptr) continue;
            auto mp = feat->map_point_.lock();
            if (mp) {
                mp->RemoveObservation(feat);
            }
        }

        CleanMap();
    }

    void Map::InsertKeyFrame(Frame::Ptr frame){


        //typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;
        //根据Map类中定义的这个unordered_map容器，我们可知插入一个关键帧需要分配给其一个容器内的索引值

        current_frame_ = frame;//Frontend和mappoint两个类里都有一个叫做current_frame_的私有成员变量，这里要注意和Frontend里面的区分
        if (keyframes_.find(frame->keyframe_id_) == keyframes_.end()) {
            //如果key存在，则find返回key对应的迭代器，如果key不存在，则find返回unordered_map::end。
            //因此当keyframes_.find(frame->keyframe_id_) == keyframes_.end()时，则说明这个要插入的关键帧在容器内原先不存在，需插入
            keyframes_.insert(make_pair(frame->keyframe_id_, frame));

            //插入原先不存在的一个关键帧就是在时间上插入了一个最新的关键帧，因此这个关键帧应该放入active_keyframes中
            active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        } else {//找到了这个关键帧编号，则以当前要插入的关键帧对原有内容覆盖
            keyframes_[frame->keyframe_id_] = frame;
            active_keyframes_[frame->keyframe_id_] = frame;
        }

        if (active_keyframes_.size() > num_active_keyframes_) {
            RemoveOldKeyframe();
        }


    }

    void Map::InsertMapPoint(MapPoint::Ptr map_point) {//与插入关键帧同理
        if (landmarks_.find(map_point->id_) == landmarks_.end()) {
            landmarks_.insert(make_pair(map_point->id_, map_point));
            active_landmarks_.insert(make_pair(map_point->id_, map_point));
        } else {
            landmarks_[map_point->id_] = map_point;
            active_landmarks_[map_point->id_] = map_point;
        }
    }
}