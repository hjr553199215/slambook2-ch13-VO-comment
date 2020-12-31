///针对mappoint.h中涉及的各类函数进行定义实现


#include "myslam/mappoint.h"
#include "myslam/feature.h"

namespace  myslam {

    MapPoint::MapPoint(long id, Vec3 position) : id_(id), pos_(position) {}

    MapPoint::Ptr MapPoint::CreateNewMappoint() {
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint(new MapPoint);
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }

    void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        for (auto iter = observations_.begin(); iter != observations_.end();
             iter++) {
            if (iter->lock() == feat) {
                observations_.erase(iter);//地图点从特征观测中删除iter指向的特征
                feat->map_point_.reset();//iter此时和feat指向的是同一个特征对象，对地图点而言，这个特征不再是它的观测，
                                        // 那么对特征而言，这个地图点也不再是它对应的地图点，也需要对feature对象删除map_point_
                observed_times_--;
                break; //找到要删除的feat后，将其从观测中删除，然后跳出循环，因为remove工作已完成
            }
        }
    }

}
