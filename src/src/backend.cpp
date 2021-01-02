#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

// 一般编程中都需要先检查一个条件才进入等待环节，因此在中间有一个检查时段，检查条件的时候是不安全的，需要lock
namespace myslam{

    Backend::Backend() {
        backend_running_.store(true);
        backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));//类成员函数需要绑定该类的指针
        //bind函数的用法和详细参考： https://www.cnblogs.com/jialin0x7c9/p/12219239.html
        //this指针的用法和详解参考： http://c.biancheng.net/view/2226.html
    }


    void Backend::UpdateMap() {
        std::unique_lock<std::mutex> lock(data_mutex_);//没有defer_lock的话创建就会自动上锁了
        //std::unique_lock:  https://murphypei.github.io/blog/2019/04/cpp-concurrent-2.html
        //std::unique_lock:  https://cloud.tencent.com/developer/article/1583807
        map_update_.notify_one(); //随机唤醒一个wait的线程
    }


    void Backend::Stop() {
        backend_running_.store(false);//replace the contained value with "parameter" 这里的parameter就是false
        map_update_.notify_one();
        backend_thread_.join();
    }

    void Backend::BackendLoop() {
        while (backend_running_.load()) {///load()   Read contained value
            std::unique_lock<std::mutex> lock(data_mutex_);
            map_update_.wait(lock);
            // wait():一般编程中都需要先检查一个条件才进入等待环节，因此在中间有一个检查时段，检查条件的时候是不安全的，需要lock
            //被notify_one唤醒后，wait() 函数也会自动调用 data_mutex_.lock()，使得data_mutex_恢复到上锁状态
            /// 后端仅优化激活的Frames和Landmarks
            Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
            Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
            Optimize(active_kfs, active_landmarks);
        }
    }


    void Backend::Optimize(Map::KeyframesType &keyframes, Map::LandmarksType &landmarks)
    {
        //优化器构造可以参照： https://www.cnblogs.com/CV-life/p/10286037.html
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> block;//对于二元边来说，这里的6,3是两个顶点的维度
        //具体的先后顺序是库内写死的，第一个是pose 第二个是point
        //g2o::BlockSolver_6_3 可以整体代替g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>
        typedef g2o::LinearSolverCSparse<block::PoseMatrixType> LinearSolverType;
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<block>(g2o::make_unique<LinearSolverType>()));
        //创建稀疏优化器
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(true);//打开调试输出



        std::map<unsigned long, VertexPose *> vertices;

        unsigned long max_kf_id = 0;
        for (auto &keyframe : keyframes)
        {
            auto kf = keyframe.second;
            VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
            vertex_pose->setId(kf->keyframe_id_);
            vertex_pose->setEstimate(kf->Pose());
            optimizer.addVertex(vertex_pose);

            if (kf->keyframe_id_ > max_kf_id) {
                max_kf_id = kf->keyframe_id_;
            }

            vertices.insert({kf->keyframe_id_, vertex_pose});
        }


        std::map<unsigned long, VertexXYZ *> vertices_landmarks;

        // K 和左右外参
        Mat33 K = cam_left_->K();
        SE3 left_ext = cam_left_->pose();
        SE3 right_ext = cam_right_->pose();


        // edges
        int index = 1;
        double chi2_th = 5.991;  // robust kernel 阈值
        std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

        for (auto &landmark : landmarks)
        {
            if (landmark.second->is_outlier_) continue;
            unsigned long landmark_id = landmark.second->id_;
            auto observations = landmark.second->GetObs();
            for (auto &obs : observations)
            {
                if (obs.lock() == nullptr) continue;
                auto feat = obs.lock();
                if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

                auto frame = feat->frame_.lock();
                EdgeProjection *edge = nullptr;

                if (feat->is_on_left_image_) {
                    edge = new EdgeProjection(K, left_ext);
                } else {
                    edge = new EdgeProjection(K, right_ext);
                }

                // 如果landmark还没有被加入优化，则新加一个顶点
                if (vertices_landmarks.find(landmark_id) == vertices_landmarks.end())
                {
                    VertexXYZ *v = new VertexXYZ();
                    v->setEstimate(landmark.second->Pos());
                    v->setId(landmark_id + max_kf_id + 1);//这里就看出max_kf_id有啥用了
                    v->setMarginalized(true); //是否边缘化,以便稀疏化求解
                    vertices_landmarks.insert({landmark_id, v});
                    optimizer.addVertex(v);
                }

                edge->setId(index);
                edge->setVertex(0, vertices.at(frame->keyframe_id_));    // pose
                //dynamic_cast<VertexPose *> ( optimizer.vertex ( frame->keyframe_id_) )
                edge->setVertex(1, vertices_landmarks.at(landmark_id));  // landmark
                //dynamic_cast<VertexXYZ *> ( optimizer.vertex ( landmark_id + max_kf_id + 1) )
                edge->setMeasurement(toVec2(feat->position_.pt));
                edge->setInformation(Mat22::Identity());//e转置*信息矩阵*e,所以由此可以看出误差向量为n×1,则信息矩阵为n×n
                auto rk = new g2o::RobustKernelHuber();
                //g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber();
                rk->setDelta(chi2_th);
                //设置鲁棒核函数，之所以要设置鲁棒核函数是为了平衡误差，不让二范数的误差增加的过快。
                // 鲁棒核函数里要自己设置delta值，
                // 这个delta值是，当误差的绝对值小于等于它的时候，误差函数不变。否则误差函数根据相应的鲁棒核函数发生变化。
                edge->setRobustKernel(rk);
                edges_and_features.insert({edge, feat});

                optimizer.addEdge(edge);

                index++;
            }
        }

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        int cnt_outlier = 0, cnt_inlier = 0;
        int iteration = 0;
        while (iteration < 5)
        {
            cnt_outlier = 0;
            cnt_inlier = 0;

            for (auto &ef : edges_and_features) {
                if (ef.first->chi2() > chi2_th) {//这里是误差大于阈值的意思吗？
                    cnt_outlier++;
                } else {
                    cnt_inlier++;
                }
            }

            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            if (inlier_ratio > 0.5) {
                break;
            } else {
                chi2_th *= 2;
                iteration++;
            }
        }


        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                ef.second->is_outlier_ = true;
                // remove the observation
                ef.second->map_point_.lock()->RemoveObservation(ef.second);
            } else {
                ef.second->is_outlier_ = false;
            }
        }

        LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
                  << cnt_inlier;

        // Set pose and lanrmark position
        for (auto &v : vertices) {
            keyframes.at(v.first)->SetPose(v.second->estimate());
        }
        for (auto &v : vertices_landmarks) {
            landmarks.at(v.first)->SetPos(v.second->estimate());
        }

    }

}