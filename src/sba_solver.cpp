#include <front_end/sba_solver.hpp>


stereoVertex::stereoVertex() : g2o::BaseVertex<6,g2o::SE3Quat>()
{
    Pl.setZero();
    Pr.setZero();///=cv::Mat::zeros(3,4,CV_64F); 
}


bool stereoVertex::read(std::istream& is)
{
    g2o::Vector7d p;
    is >> p[0] >> p[1] >> p[2] >> p[3] >> p[4] >> p[5] >> p[6];
    
    _estimate.fromVector(p);
    return true;
}

bool stereoVertex::write(std::ostream& os) const
{
    g2o::Vector7d p = _estimate.toVector();
    os << p[0] << " " << p[1] << " " << p[2]
    << p[3] << p[4] << p[5] << p[6] ;
    return os.good();
}


void stereoVertex::oplusImpl(const double* update)
{
   Eigen::Quaterniond newR(update[1],update[2],update[3],update[0]);
   Eigen::Vector3d newT; 
   newT(0,0)=update[4];;
   newT(1,0)=update[5];
   newT(2,0)=update[6];
   _estimate*g2o::SE3Quat(newR,newT);
}
/////////////////////////////////////////////////////////////////

landmarkEdge::landmarkEdge() : BaseBinaryEdge<4, Eigen::Vector4d, stereoVertex, g2o::VertexPointXYZ> ()
{

}


bool landmarkEdge::read(std::istream& is)
{

}


bool landmarkEdge::write(std::ostream& os) const
{
    
}

void landmarkEdge::computeError()
{
    const stereoVertex* cam=static_cast<const stereoVertex*>(_vertices[0]);


    const g2o::VertexPointXYZ* landmark = static_cast<const g2o::VertexPointXYZ*>(_vertices[1]);
    
    double landmarkData[3];
    landmark->getEstimateData(&landmarkData[0]);
    Eigen::Vector4d X;
    X(0,0)=landmarkData[0];
    X(1,0)=landmarkData[1];
    X(2,0)=landmarkData[2];
    X(3,0)=1;

    Eigen::Vector3d Left,Right;
    Left=cam->Pl*X;
    Left/=Left(2,0);
    Right=cam->Pr*X;
    Right/=Right(2,0);

    Eigen::Vector4d predicted,error;
    predicted(0,0)=Left(0,0);
    predicted(1,0)=Left(1,0);
    predicted(2,0)=Right(0,0);
    predicted(3,0)=Right(1,0);

    predicted-=_measurement;
    _error = predicted;
}

void landmarkEdge::setMeasurement(const Eigen::Vector4d &m)
{
    _measurement=m;
}