#pragma once
#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"


//Config类，这个类从名称上就很容易看出来是读取配置文件的，我们会有一个yaml文件或者其他类型的文件用来存放一些系统的超参数
//Config类提供了统一的函数用来确定配置文件的位置，并从中读取相应参数。
namespace myslam{


    class Config{
    private:
        static std::shared_ptr<Config> config_;//将智能指针申明为静态的，它就只与类有关了，
                                              // static 成员变量既可以通过对象来访问，也可以通过类来访问，
    //静态数据成员被 类 的所有对象所共享，包括该类派生类的对象。即派生类对象与基类对象共享基类的静态数据成员。
    //也就是说无论有多少个对象，它们实际上操作的都是同一个静态成员变量，即静态成员变量对所有对象来说是共享的


        cv::FileStorage file_;

        Config() {}  // private constructor makes a singleton
        //类的构造函数一般是public的，但是也可以是private的。构造函数为私有的类有这样的特点：
        //<1>不能实例化：因为实例化时类外部无法访问其内部的私有的构造函数；
        //<2>不能继承：同<1>；
    public:
        ~Config();  // close the file when deconstructing

        // set a new config file
        static bool SetParameterFile(const std::string &filename);

        // access the parameter values
        template <typename T>  //模板类T，这个定义方式template <typename T>是c++规定死的东西
        //T就相当于是一个万能的类型，可以是int,double,float......等各种，在具体调用的时候可以用各种不同的数据类型，避免为了不同的
        //数据类型去重复写函数实现


        //静态成员函数主要为了调用方便，不需要生成对象就能调用，它跟类的实例无关，只跟类有关，不需要this指针。
        //从OOA/OOD的角度考虑，一切不需要实例化就可以有确定行为方式的函数都应该设计成静态的。
        static T Get(const std::string &key) {
            return T(Config::config_->file_[key]);
        }
    };

}

#endif