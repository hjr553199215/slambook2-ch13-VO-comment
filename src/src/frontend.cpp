#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"//
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"//
#include "myslam/viewer.h"

namespace myslam {

    Frontend::Frontend() {
        gftt_ = cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
        num_features_init_ = Config::Get<int>("num_features_init");
        num_features_ = Config::Get<int>("num_features");
    }


    bool Frontend::AddFrame(myslam::Frame::Ptr frame) {

        current_frame_ = frame;
        switch (status_)//status_是前端Frontend的类成员，所以这里判断的是整个前端的状态
        {
            case FrontendStatus::INITING:
                StereoInit()//这里的StereoInit应该是一个bool函数
                break;
            case FrontendStatus::TRACKING_GOOD://不加break就不会跳出switch结构，不管后面case的条件是否符合都将会执行，
                // 直到遇到第一个break才会跳出switch结构
                //此处的TRACKING_GOOD和TRACKING_BAD并非跟踪失败或者跟踪成功的含义，这里的good和bad只是跟踪时条件好坏的区别
                //当跟踪到的特征点数目充足时就是good，当特征点数目不足时就是bad，但不论good还是bad，都只是一个条件恶劣与否的问题，并不涉及失败
                //当特征点少到不行的时候，就已经不是good和bad的问题了，太少的时候我们就认为跟踪丢了，设置为LOST，这个lost可以理解为跟踪失败
                //所以，lost之后需要reset系统
            case FrontendStatus::TRACKING_BAD:
                Track();
                break;
            case FrontendStatus::LOST:
                Reset();
                break;
        }
        last_frame_ = current_frame_;
        return ture;
    }

//根据上面的Addframe函数，我们应当在后续实现StereoInit，Track和Reset各个函数


//先来StereoInit
    bool StereoInit() {
        int num_features_left = DetectFeatures();//一个frame其实就是一个时间点，里面同时含有左，右目的图像。这一步在提取左目特征,
        //通常在左目当中提取特征时特征点数量是一定能保证的。
        int num_coor_features = FindFeaturesInRight();//根据左目特征在右目中找对应，虽然左目提取时特征点数量能够保证，但匹配过程则无法确保
        //能够在右目图像中为所有的左目特征都找到对应，所以这一步最后找到的对应特征数目不一定满足初始化条件

        if (num_coor_features < num_features_init_) {//对应数目不足，无法初始化
            return false;
        }

        bool build_map_success = BuildInitMap();//初始化成功，则开始建立初始的地图

        if (build_map_success) {    //初始地图建立成功
            status_ = FrontendStatus::TRACKING_GOOD;   //前端状态可有INITING转变为TRACKING_GOOD
            if (viewer_) {   //可视化器是否开启
                viewer_->AddCurrentFrame(current_frame_);  //将当前帧加入可视化器
                viewer_->UpdateMap();//更新可视化内容以显示当前帧,

                //为什么Track()函数里面只有AddCurrentFrame(current_frame_)，没有UpdateMap()呢？
                //还需后续读一读AddCurrentFrame(current_frame_)和UpdateMap()的具体源码实现。
            }
            return true;
        }

        //如果初始地图没有建立成功，则会跳过上述if段落，则还是没能完成初始化工作，返回false
        return false;
    }

//假如初始化成功后，此时前端状态已经变为case FrontendStatus::TRACKING_GOOD:，再来一帧之后，则会执行Track()

    bool Track() {
        //在执行Track之前，需要明白，Track究竟在做一件什么事情,Track是当前帧和上一帧之间进行的匹配，而初始化是某一帧左右目（双目）之间进行的匹配
        //这个要分清楚
        //先看last_frame_是不是正常存在的，需要先对其进行一个判断
        if (last_frame_)//判断last_frame_是不是空指针，如果是空指针，说明这个指针没有指到正确的上一帧所在的内存地址，则不执行后续语句，等同于false
        {
            current_frame_->SetPose(relative_motion_ * last_frame_->Pose());//当前帧pose等于上一帧pose加上一个帧间pose
            //world----last_frame*last_frame------current_frame = world----current_frame
        }

        int num_track_last = TrackLastFrame();  //从上一帧跟踪到当前帧，跟踪到了多少个特征，num_track_last可理解为从上一帧跟踪过来到当前帧
        //跟踪到了的特征数目。
        tracking_inliers_ = EstimateCurrentPose(); //因为前面已经知道了一个current_pose，所以可以根据这个pose初值判断inliers和outliers
        //(重投影)，同时根据这个函数的名字，我猜测后面还有对当前帧pose的精化。
        //接下来根据跟踪到的内点的匹配数目，可以分类进行后续操作
        //(------lost--------)<num_features_tracking_bad_<(-------bad---------)<num_features_tracking_<(--------good--------)
        if (tracking_inliers_ > num_features_tracking_) {
            // tracking good
            status_ = FrontendStatus::TRACKING_GOOD;
        } else if (tracking_inliers_ > num_features_tracking_bad_) {
            // tracking bad
            status_ = FrontendStatus::TRACKING_BAD;
        } else {
            // lost
            status_ = FrontendStatus::LOST;
        }

        InsertKeyframe();//根据当前的tracking_inliers_判断其是否为关键帧,
        // 当少于特定数目的点时，则可认为是一个关键帧，否则的话就不是，返回false，但我个人理解这个判断是否为关键帧的阈值不能太少，否则就会lost
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();
        //看到这里可能会有些困惑，怎么又出来一个relative_motion呢？下面我作以详细介绍
        /*
         *假设现在有3个连续帧A，B，C其中在对C进行Track()函数操作时，首先选用的那个relative_motion其实是A和B之间精准的relative_motion,
         * 为什么可以这么做呢，因为我们可以认为A，B，C这三帧之间的采样间隔是比较短的，或者说A-B之间的相对运动和B-C之间的相对运动之间隔的时间很短
         * 基于这种小运动假设，我们可以认为A-B和B-C的相对运动其实差不多，可以拿A-B的相对运动作为B-C相对运动的一个初值来做inliers和outliers的
         * 判别，所以一开始求出的那个current_frame_pose其实是一个基于假设的粗略初值，只能拿来做一个outliers筛查，这也解释了为什么后面又求了
         * 一个relative_motion,这个时候的relative_motion由于前面EstimateCurrentPose()函数的存在，已经是精准的B-C相对运动了。
         */
        if (viewer_) viewer_->AddCurrentFrame(current_frame_);//可视化
        return true;

    }

//针对前面的status_,已经将INITING，GOOD，BAD三种情况下对应的Stereoinit和Track函数都说明了，接下来说明Reset函数
    bool Reset() {
        LOG(INFO) << "Reset is not implemented. ";
        status_ = FrontendStatus::INITING;//前端状态重置

        return true;
        /*
         * 在需要插入日志的地方调用LOG(TYPE)<<”yourinfo”;即可。your info表示你要输入到日志文件中的信息。
         */
//高博在这里并没有做Reset的实现，我们可以在后续读完程序后给他补上

    }

//三个上层级函数已经实现，接下来对stereoInit，Track，Reset三个函数中的一些细节函数再作以补充实现。
//首先对StereoInit函数中的DetectFeatures()，FindFeaturesInRight()，BuildInitMap()三个函数做实现，可视化模块放在后面统一实现
    int DetectFeatures() {
        cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);//掩膜，灰度图，同时可以看出，DetectFeatures是对左目图像的操作
        for (auto &feat : current_frame_->features_left_) {
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                          feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);//在已有的特征附近一个矩形区域内将掩膜值设为0
        }                                                                  //即在这个矩形区域中不提取特征了，保持均匀性，并避免重复
        //auto语法讲解：auto可以根据初始化值进行自动类型推断，在这里auto&定义了feat，初始化值就是后面的current_frame_->features_left_
        //auto&在自动类型推断完成定义的同时构成了引用类型，也就是说feat的改变将同步影响current_frame_->features_left_中的元素
        //如果单单用一个auto就不会有这种同步效果
        std::vector <cv::KeyPoint> keypoints; //关键点容器
        gftt_->detect(current_frame_->left_img_, keypoints, mask);
        //detect函数，第三个参数是用来指定特征点选取区域的，一个和原图像同尺寸的掩膜，其中非0区域代表detect函数感兴趣的提取区域，相当于为
        //detect函数明确了提取的大致位置
        int cnt_detected = 0;//检测到的特征计数
        for (auto &kp : keypoints) {
            current_frame_->features_left_.push_back(
                    Feature::Ptr(new Feature(current_frame_, kp)));
            cnt_detected++;
        }

        LOG(INFO) << "Detect " << cnt_detected << " new features";
        return cnt_detected;
    }

/*
int DetectFeatures(){
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);//掩膜，同时由这句可以看出，DetectFeatures是对左目图像的操作


    std::vector<cv::KeyPoint> keypoints; //关键点容器
    gftt_->detect(current_frame_->left_img_, keypoints);

    int cnt_detected = 0;//检测到的特征计数
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
                Feature::Ptr(new Feature(current_frame_, kp)));
        //这一句牵涉到两个知识点，第一个时new的用法，第二个是智能指针初始化赋值方法
        //new Feature(.....)这个整体应当是一个地址或者说指针含义，我们应当将这个地址赋值给一个Feature类型的智能指针，
        // 假如我们有一个指针int *p , 那么p这个变量就是一个指针，而单纯的使用p实则就是用它内部存储的地址，所以用p的时候可以说是用指针，
        //也可以说是用地址
        // Feature::Ptr(....)是一种初始化方法，就像个构造函数 只不过这里没有声明具体的存放变量，而是构造出来后直接push_back到容器里面，
        //所以可以理解为vector里面的一个单元就是具体的存放变量。
        cnt_detected++;
    }

    for (auto &feat : current_frame_->features_left_){
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }//感觉这里就是为了画个矩形把点标注出来而已。

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}*/

//找到左目图像的feature之后，就在右目里面找对应值
    int FindFeaturesInRight() {
        std::vector <cv::Point2f> kps_left, kps_right;//定义两个存储关键点坐标的vector容器

        //遍历左目特征的关键点
        for (auto &kp : current_frame_->features_left_) {
            kps_left.push_back(kp->position_.pt);
            auto mp = kp->map_point_.lock();//通过weak_ptr的lock()函数实现对地图点shared_ptr智能指针的复制，并赋予mp
            if (mp) {   //如果赋予成功，则mp是一个非空的智能指针，则执行if语段
                // use projected points as initial guess
                auto px =
                        camera_right_->world2pixel(mp->pos_, current_frame_->Pose());

                kps_right.push_back(cv::Point2f(px[0], px[1]));
            } else {   //如果指针为空则执行else语段
                // use same pixel in left iamge
                kps_right.push_back(kp->position_.pt);
            }
        }

        //进行光流跟踪，从这条opencv光流跟踪语句我们就可以知道，前面遍历左目特征关键点是为了给光流跟踪提供一个右目初始值
        std::vector <uchar> status; //光流跟踪成功与否的状态向量（无符号字符），成功则为1,否则为0
        cv::calcOpticalFlowPyrLK(
                current_frame_->left_img_, current_frame_->right_img_, kps_left,
                kps_right, status
        cv::Size(11, 11), 3,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                                 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);
        //OPTFLOW_USE_INITIAL_FLOW使用初始估计，存储在nextPts中;如果未设置标志，则将prevPts复制到nextPts并将其视为初始估计。

        int num_good_pts = 0;//右目中光流跟踪成功的点
        for (size_t i = 0; i < status.size(); ++i) {
            /*size_t的一些说明，（感觉这里用size_t意义不大）
             * 它是一种“整型”类型，里面保存的是一个整数，就像int, long那样。这种整数用来记录一个大小(size)。size_t的全称应该是size type，就是说“一种用来记录大小的数据类型”。
    通常我们用sizeof(XXX)操作，这个操作所得到的结果就是size_t类型。因为size_t类型的数据其实是保存了一个整数，
             所以它也可以做加减乘除，也可以转化为int并赋值给int类型的变量。
             */
            if (status[i]) {
                cv::KeyPoint kp(kps_right[i], 7);
                //上述KeyPoint构造函数中7代表着关键点直径
                Feature::Ptr feat(new Feature(current_frame_, kp));
                feat->is_on_left_image_ = false;
                current_frame_->features_right_.push_back(feat);
                num_good_pts++;
            } else {
                current_frame_->features_right_.push_back(nullptr);//光流跟踪没找到的特征，就在features_right_里面填空指针
            }
        }

        LOG(INFO) << "Find " << num_good_pts << " in the right image.";
        return num_good_pts;
    }

//现在左目图像的特征提取出来了，并根据左目图像的特征对右目图像做了特征的光流跟踪，找到了对应值，当对应数目满足阈值条件时，我们可以开始建立
//初始地图
    bool BuildInitMap() {
        std::vector <SE3> poses{camera_left_->pose(), camera_right_->pose()};
        //构造一个存储SE3的vector，里面初始化就放两个pose，一个左目pose，一个右目pose，看到这里应该记得，对Frame也有一个pose，Frame里面的
        //pose描述了固定坐标系（世界坐标系）和某一帧间的位姿变化，这个固定坐标系可能是第一帧的左目，可能是右目，也可能是左右目中间。
        size_t cnt_init_landmarks = 0; //初始化的路标数目

        //遍历左目
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            //对于左目的每一个feature，我们必须在右目找到对应的feature，才能继续三角化，不然就跳过
            //对于右目feature容器，如果是成功跟踪到了就是一个指向Feature的指针，否则就是个空指针，我们需要跳过跟踪失败的空指针
            if (current_frame_->features_right_[i] == nullptr) continue;

            //对于左右目配对成功的点，三角化它
            std::vector <Vec3> points{//左目右目的像素坐标都转为相机坐标，存到一个vector里面，下一次循环中将重置一个含有两个元素的新vector
                    //将配对点中的左目像素坐标转换到相机（归一化）坐标
                    camera_left_->pixel2camera(
                            Vec2(current_frame_->features_left_[i]->position_.pt.x,
                                 current_frame_->features_left_[i]->position_.pt.y)),
                    //将配对点中的右目像素坐标转换到相机（归一化）坐标
                    camera_right_->pixel2camera(
                            Vec2(current_frame_->features_right_[i]->position_.pt.x,
                                 current_frame_->features_right_[i]->position_.pt.y))

            };
            Vec3 pworld = Vec3::Zero(); //建立一个存储3D世界坐标的VEC3，3D向量

            //正式三角化
            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                //根据前面存放的左右目相机pose和对应点相机坐标points进行三角化，得到对应地图点的深度，构造出地图点pworld
                //需要对pworld进行判断，看其深度是否大于0, pworld[2]即是其深度。


                //创建一个MapPoint类对象用来承载三角化出的世界坐标pworld
                auto new_map_point = MapPoint::CreateNewMappoint();
                new_map_point->SetPos(pworld);

                //为这个地图点添加观测量，这个地图点对应到了当前帧（应有帧ID）的左目图像特征中的第i个以及右目图像特征中的第i个
                new_map_point->AddObservation(current_frame_->features_left_[i]);
                new_map_point->AddObservation(current_frame_->features_right_[i]);

                //上两句是为地图点添加观测，这两句就是为特征类Feature对象填写地图点成员
                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;

                //初始化成功的landmark或者说地图点数目加1
                cnt_init_landmarks++;

                //对Map类对象来说，地图里面应当多了一个地图点，所以要将这个地图点加到地图中去
                map_->InsertMapPoint(new_map_point);

            }
        }
        //当前帧能够进入初始化说明已经满足了初始化所需的帧特征数量，作为初始化帧，可看做开始的第一帧，所以应当是一个关键帧
        current_frame_->SetKeyFrame();

        //对Map类对象来说，地图里面应当多了一个关键帧，所以要将这个关键帧加到地图中去
        map_->InsertKeyFrame(current_frame_);

        //关键帧插入，后端需要对新纳入的关键帧进行优化处理
        backend_->UpdateMap();


        //向日志输入消息
        LOG(INFO) << "Initial map created with " << cnt_init_landmarks
                  << " map points";

        return true;
    }


//接下来对Track函数中的TrackLastFrame()，EstimateCurrentPose()，InsertKeyframe()三个函数做实现，可视化模块放在后面统一实现

//TrackLastFrame()从上一帧跟踪到当前帧

    int TrackLastFrame() {
        //该函数的实现其实非常像FindFeaturesInRight(),不同的是一个在左右目之间找，另一个在前后帧之间找
        std::vector <cv::Point2f> kps_last, kps_current;//定义两个存储关键点坐标的vector容器

        for (auto &kp : last_frame_->features_left_) {
            //遍历上一帧中的所有左目特征
            if (kp->map_point_.lock()) {
                //判断该特征有没有构建出相应的地图点
                //这里需要注意了，对于左目图像来说，我们可以将其用于估计相机pose，但是不一定左目图像中的每一个点都有mappoint
                //MapPoint的形成是需要左目和同一帧的右目中构成对应关系才可以，有些左目中的feature在右目中没有配对，就没有Mappoint
                //但是没有Mappoint却不代表这个点是一个outlier
                // use project point
                auto mp = kp->map_point_.lock();
                auto px =
                        camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(cv::Point2f(px[0], px[1]));
            } else {
                kps_last.push_back(kp->position_.pt);
                //没有MapPoint就没有初始化猜测值，那么光流搜索的起点就是上一帧中点的像素位置
                kps_current.push_back(kp->position_.pt);
            }
        }

        std::vector <uchar> status;
        cv::calcOpticalFlowPyrLK(
                last_frame_->left_img_, current_frame_->left_img_, kps_last,
                kps_current, status, cv::Size(11, 11), 3,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                                 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;

        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {//status[i]=true则说明跟踪成功有对应点，false则跟踪失败没找到对应点
                cv::KeyPoint kp(kps_current[i], 7);
                Feature::Ptr feature(new Feature(current_frame_, kp));


                feature->map_point_ = last_frame_->features_left_[i]->map_point_;
                //这个时候这里面的map_point_有可能之前没在右目中找到对应点，所以其对应的map_point_未曾被初始化或者赋值，也可以这么写吗？
                //针对于这种情况，应该只是保证了feature->map_point_和last_frame_->features_left_[i]->map_point_之间的相同性，
                // 但却并没有关注last_frame_->features_left_[i]->map_point_是否被初始化或者赋值。

                current_frame_->features_left_.push_back(feature);
                num_good_pts++;
            }
        }

        LOG(INFO) << "Find " << num_good_pts << " in the last image.";
        return num_good_pts;

    }

//跟踪成功之后可以求解精准的Current——Pose

//EstimateCurrentPose()求解精准的CurrentPose

    int EstimateCurrentPose() {
        //利用g2o来进行优化求解，先进行优化求解器配置
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverDense <BlockSolverType::PoseMatrixType>
                LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<BlockSolverType>(
                        g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        //明确一下这个过程中我们要优化的变量是什么，这里我们认为地图的3D位置是不用再去优化的（想优化也可以），所以这个过程中只需要优化R，t
        //而R，t可以统一为一个变量SE3，所以这个两帧间的SE3就是我们的优化对象，也就是g2o中的一个顶点（一个变量）


        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose());//设定初值
        optimizer.addVertex(vertex_pose);//添加顶点

        // K
        Mat33 K = camera_left_->K();//左目内参矩阵

        //设定边
        int index = 1; //建立索引
        std::vector < EdgeProjectionPoseOnly * > edges;  //建立边的容器，边类型为EdgeProjectionPoseOnly
        std::vector <Feature::Ptr> features; //建立一个特征容器

        //因为有多个投影点构成多个方程，所以有多个边，一对特征就有一个重投影误差项，就是一条边。

        //建立并添加边
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            auto mp = current_frame_->features_left_[i]->map_point_.lock();
            if (mp)//这里就涉及到前面在TrackLastFrame()函数里面提到的，有些特征虽然被跟踪到了，但是并没有受到三角化，即没有map_point_
                //这里便对feature有没有map_point_进行判断，有则可以往下进行重投影，没有则不行，因为重投影需要点的3D位置
            {
                features.push_back(current_frame_->features_left_[i]);
                EdgeProjectionPoseOnly *edge =
                        new EdgeProjectionPoseOnly(mp->pos_, K);
                edge->setId(index);
                edge->setVertex(0, vertex_pose);
                edge->setMeasurement(
                        toVec2(current_frame_->features_left_[i]->position_.pt));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge);
                optimizer.addEdge(edge);
                index++;
            }
        }

        // estimate the Pose the determine the outliers
        const double chi2_th = 5.991;
        int cnt_outlier = 0;
        for (int iteration = 0; iteration < 4; ++iteration) {
            //总共优化了40遍，以10遍为一个优化周期，对outlier进行一次判断并舍弃掉outlier的边，随后再进行下一个10步优化
            vertex_pose->setEstimate(current_frame_->Pose());//每次优化的初值都设定为current_frame_->Pose()，
            // 但每次涉及的特征都不一样，所以每次的重投影误差都不一样，
            // 就有可能发现新的outlier，这是一个不断筛查,删除,精化的过程
            optimizer.initializeOptimization();
            optimizer.optimize(10);
            cnt_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i) {
                auto e = edges[i];
                if (features[i]->is_outlier_) {
                    e->computeError();
                }
                if (e->chi2() > chi2_th) {
                    features[i]->is_outlier_ = true;
                    e->setLevel(1);
                    //这里每个边都有一个level的概念，默认情况下，g2o只处理level=0的边，在orbslam中，
                    // 如果确定某个边的重投影误差过大，则把level设置为1，也就是舍弃这个边对于整个优化的影响
                    cnt_outlier++;
                } else {
                    features[i]->is_outlier_ = false;
                    e->setLevel(0);
                };

                if (iteration == 2) {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
                  << features.size() - cnt_outlier;
        // Set pose and outlier
        current_frame_->SetPose(vertex_pose->estimate());

        LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

        for (auto &feat : features) {
            if (feat->is_outlier_) {
                feat->map_point_.reset();//弱指针自带的操作函数reset，作用是将指针置空
                feat->is_outlier_ = false;  // maybe we can still use it in future？？？？？？这里这么搞是怎么个意思
            }
        }
        return features.size() - cnt_outlier;//inliers
    }

//在完成前后帧的跟踪和pose估计后，我们需要对新来的每一帧进行关键帧判别，看它是不是一个关键帧，这里就需要用到InsertKeyframe函数

    bool InsertKeyframe() {
        //d当跟踪到的特征数目小于阈值时认为运动已有足够大的空间，时间幅度，可视做一个新的关键帧
        if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
            // still have enough features, don't insert keyframe
            return false;
        }

        // current frame is a new keyframe
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);

        LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
                  << current_frame_->keyframe_id_;


        SetObservationsForKeyFrame();//
        DetectFeatures();  // detect new features , 如果是关键帧才能执行到这一步，是关键帧的话其跟踪到的内点数目就会相应不足，需要补充

        // track in right image
        FindFeaturesInRight();
        // triangulate map points
        TriangulateNewPoints();
        // update backend because we have a new keyframe
        backend_->UpdateMap();

        if (viewer_) viewer_->UpdateMap();

        return true;

    }

//在InsertKeyFrame函数中出现了一个三角化步骤，这是因为当一个新的关键帧到来后，我们势必需要补充一系列新的特征点，
// 此时则需要像建立初始地图一样，对这些新加入的特征点进行三角化，求其3D位置

    int TriangulateNewPoints() {
        //这个函数其实与BuildInitMap差不多

        std::vector <SE3> poses{camera_left_->pose(), camera_right_->pose()};
        SE3 current_pose_Twc = current_frame_->Pose().inverse();  //current_frame_->Pose()是从世界到相机,逆就是从相机到世界
        int cnt_triangulated_pts = 0;  //三角化成功的点的数目

        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {//遍历新的关键帧（左目）内的所有特征点，
            // 和右目对应点做三角化
            if (current_frame_->features_left_[i]->map_point_.expired() &&
                current_frame_->features_right_[i] != nullptr) {
                // 左图的特征点未关联地图点且存在右图匹配点，则可以尝试三角化
                std::vector <Vec3> points{  //将匹配的像素点从像素坐标转化到相机坐标下
                        camera_left_->pixel2camera(
                                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                                     current_frame_->features_left_[i]->position_.pt.y)),
                        camera_right_->pixel2camera(
                                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                                     current_frame_->features_right_[i]->position_.pt.y))};
            }

            Vec3 pworld = Vec3::Zero();//申明相机坐标系下点的3D位置坐标

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {//如果三角化流程运行成功且得到的深度值pworld[2]大于0有意义
                //创建并设立地图点
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;//从相机坐标系转到世界坐标系
                new_map_point->SetPos(pworld);

                //为地图点类添加特征成员变量
                new_map_point->AddObservation(
                        current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                        current_frame_->features_right_[i]);

                //为特征类添加地图点成员变量
                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;

                //既然有了一个新的地图点mappoint，那就应当更新一下地图（类），向地图类的对象中添加地图点。
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }


        }
        LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;//新添加了多少个地图点（landmarks）
        return cnt_triangulated_pts;

    }

    void Frontend::SetObservationsForKeyFrame() {  //查找当前帧中的特征，看是否对应已有的地图点，若对应则为地图点添加当前帧内的特征观测
        //若不对应则不做操作，跳过即可
        for (auto &feat : current_frame_->features_left_) {
            auto mp = feat->map_point_.lock();
            if (mp) mp->AddObservation(feat);
        }
    }

}