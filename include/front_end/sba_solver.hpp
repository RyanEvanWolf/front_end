#ifndef FRONT_END_SBA_SOLVER_HEADER_HPP
#define FRONT_END_SBA_SOLVER_HEADER_HPP


#include "g2o/core/base_vertex.h"
#include "g2o/types/slam3d/se3quat.h"
#include "g2o/core/base_binary_edge.h" 

#include <g2o/types/slam3d/vertex_pointxyz.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 


class stereoVertex :public g2o::BaseVertex<6,g2o::SE3Quat>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    stereoVertex();
    Eigen::Matrix<double,3,4> Pl,Pr;
    //admin
    virtual void setToOriginImpl() {
        _estimate=g2o::SE3Quat();
    }
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream &os)const ;
    virtual void oplusImpl(const double* update);
};

class landmarkEdge : public g2o::BaseBinaryEdge<4, Eigen::Vector4d, stereoVertex, g2o::VertexPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    landmarkEdge();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
    void computeError();
    void setMeasurement(const Eigen::Vector4d &m);

};

#endif