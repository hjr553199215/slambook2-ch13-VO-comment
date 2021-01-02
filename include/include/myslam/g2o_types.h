///后端中我们使用g2o库进行优化，为了适应我们的优化问题，需要单独在g2o_typrs.h中新定义出一些edge类型和vertex类型以供后用

#ifndef MYSLAM_G2O_TYPES_H
#define MYSLAM_G2O_TYPES_H

#include "myslam/common_include.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>


namespace myslam{

    class VertexPose : public g2o::BaseVertex<6, SE3>
    {   //g2o构造顶点参考： https://www.cnblogs.com/CV-life/p/10449028.html
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        //如果不使用override，当你手一抖，将foo()写成了f00()会怎么样呢？
        // 结果是编译器并不会报错，因为它并不知道你的目的是重写虚函数，而是把它当成了新的函数。
        // 如果这个虚函数很重要的话，那就会对整个程序不利。
        //所以，override的作用就出来了，它指定了子类的这个虚函数是重写的父类的，如果你名字不小心打错了的话，编译器是不会编译通过的：

        virtual void setToOriginImpl() {_estimate = SE3();}

        virtual void oplusImpl(const double *update)
        {
            Vec6 update_eigen;
            update_eigen << update[0], update[1], update[2], update[3], update[4],
                    update[5];
            _estimate = SE3::exp(update_eigen) * _estimate;
        }
        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }
    };


    class VertexXYZ : public g2o::BaseVertex<3, Vec3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() { _estimate = Vec3::Zero(); }

        virtual void oplusImpl(const double *update) {
            _estimate[0] += update[0];
            _estimate[1] += update[1];
            _estimate[2] += update[2];
        }

        virtual bool read(std::istream &in) { return true; }

        virtual bool write(std::ostream &out) const { return true; }
    };


    class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Vec2, VertexPose>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeProjectionPoseOnly(const Vec3 &pos, const Mat33 &K)
                : _pos3d(pos), _K(K) {}

        virtual void computeError() override {
            const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
            SE3 T = v->estimate();
            Vec3 pos_pixel = _K * (T * _pos3d);
            pos_pixel /= pos_pixel[2];
            _error = _measurement - pos_pixel.head<2>();
        }

        virtual void linearizeOplus() override {
            const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
            SE3 T = v->estimate();
            Vec3 pos_cam = T * _pos3d;
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Zinv = 1.0 / (Z + 1e-18);
            double Zinv2 = Zinv * Zinv;
            _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                    -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                    fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                    -fy * X * Zinv;
        }

        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }

    private:
        Vec3 _pos3d;
        Mat33 _K;
    };

    class EdgeProjection
            : public g2o::BaseBinaryEdge<2, Vec2, VertexPose, VertexXYZ> {
        //测量值维度,测量值类型,第一个顶点类型,第二个顶点类型
        ///在定义边的时候，这个参数顺序就决定了第一个是0号顶点，第二个是1号顶点，后续往边内添加顶点的时候就得遵从这个标号！！！
        //g2o边的构造参考： https://www.cnblogs.com/CV-life/archive/2019/03/13/10525579.html
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /// 构造时传入相机内外参
        EdgeProjection(const Mat33 &K, const SE3 &cam_ext) : _K(K) {
            _cam_ext = cam_ext;
        }


        virtual void computeError() override {
            const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
            const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]);
            SE3 T = v0->estimate();
            Vec3 pos_pixel = _K * (_cam_ext * (T * v1->estimate()));
            pos_pixel /= pos_pixel[2];
            _error = _measurement - pos_pixel.head<2>();
        }

        virtual void linearizeOplus() override {
            const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
            const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]);
            SE3 T = v0->estimate();
            Vec3 pw = v1->estimate();
            Vec3 pos_cam = _cam_ext * T * pw;//这里这个导数的部分貌似求得有问题,后面咱们修改一下这里
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Zinv = 1.0 / (Z + 1e-18);
            double Zinv2 = Zinv * Zinv;
            _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                    -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                    fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                    -fy * X * Zinv;

            _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) *
                               _cam_ext.rotationMatrix() * T.rotationMatrix();
        }

        virtual bool read(std::istream &in) override { return true; }

        virtual bool write(std::ostream &out) const override { return true; }

    private:
        Mat33 _K;
        SE3 _cam_ext;
    };

}

#endif  // MYSLAM_G2O_TYPES_H
