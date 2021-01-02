#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

///algorithm.h文件主要实现了一个三角化函数，其实个人感觉三角化算法不单独用头文件和cpp文件来做也是可行的

namespace myslam {

    inline bool triangulation(const std::vector <SE3> &poses, const std::vector <Vec3> points, Vec3 &pt_world) {

        MatXX A(2 * poses.size(), 4);
        VecX b(2 * poses.size());
        b.setZero();
        for (size_t i = 0; i < poses.size(); ++i) {
            Mat34 m = poses[i].matrix3x4();
            //这里的三角化手段不同于十四讲书上的，十四讲书上通过两帧之间的R,t来三角化深度值，要注意的是我们这里有三个坐标系
            //当然我们可以继续像十四讲书上那样，求出左目右目间的外参，然后通过归一化坐标求出在左目或者右目坐标系下的深度值
            //然后变换到居中坐标系下，但是我们也可以像下面这样直接求解3D点坐标，3D坐标有3个未知量
            //一对匹配点可以贡献4个方程，一个特征点贡献2个，4个方程解3个未知量，最后属于求一个超定齐次线性方程组，可以用SVD方法
            A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
            A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
        }
        auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();//将齐次坐标最后一维化成1,再取前三维，就是世界坐标

        if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {//这里的判断是怎么来的？
            //因为最后一个特征值代表着最小二乘问题的最优解大小，当svd.singularValues()[3]远小于vd.singularValues()[2]时
            //我们认为这个对应的特征向量，或者说对应的解的的确确是最小的那一个，或者说在最小这件事上这个解是当之无愧的第一，没有争议
            //这个时候我们认为这个解的质量是过关的，这个解是可靠的，是不存在歧义的。
            return true;
        }

        // 解质量不好，就是说可能存在另一个特征向量对应的特征值和这个我们选出来的解特征向量对应的特征值大小比较接近
        // ，在最小化这件事情上它们两个解的效果应该差不多，这个时候就存在歧义了，我们的SVD解就显得不可靠了，就得放弃
        return false;
    }

    // converters
    inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }
}

#endif  // MYSLAM_ALGORITHM_H