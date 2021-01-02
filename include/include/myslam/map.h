#pragma once
#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam{

    /**
 * @brief 地图
 * 和地图的交互：前端调用InsertKeyframe和InsertMapPoint插入新帧和地图点，后端维护地图的结构，判定outlier/剔除等等
 */
    class Map {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Map> Ptr;
        typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;//unordered_map 是一种容器，就像哈希表一样，
                                                                              // 前面的是key值，后面的是映射对象
        typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

        Map() {}

        /// 增加一个关键帧
        void InsertKeyFrame(Frame::Ptr frame);
        /// 增加一个地图顶点
        void InsertMapPoint(MapPoint::Ptr map_point);
         /// 获取所有地图点
        LandmarksType GetAllMapPoints() {
            std::unique_lock<std::mutex> lck(data_mutex_);//防止和InsertKeyFrame,CleanMap,RemoveOldKeyframe等函数冲突
            return landmarks_;
        }
        /// 获取所有关键帧
        KeyframesType GetAllKeyFrames() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return keyframes_;
        }

        /// 获取激活地图点
        LandmarksType GetActiveMapPoints() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_landmarks_;
        }

        /// 获取激活关键帧
        KeyframesType GetActiveKeyFrames() {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_keyframes_;
        }

        /// 清理map中观测数量为零的点
        void CleanMap();


     private:
        // 将旧的关键帧置为不活跃状态
        void RemoveOldKeyframe();

        std::mutex data_mutex_;
        LandmarksType landmarks_;         // all landmarks
        LandmarksType active_landmarks_;  // active landmarks
        KeyframesType keyframes_;         // all key-frames
        KeyframesType active_keyframes_;  // all key-frames

        Frame::Ptr current_frame_ = nullptr;

        // settings
        int num_active_keyframes_ = 7;  // 激活的关键帧数量,这里的激活就是一个窗口概念
    };


}

#endif