#include <ros/ros.h>



////////////////////////////



#include <front_end/sba_solver.hpp>


#include "g2o/config.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"


#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#if defined G2O_HAVE_CHOLMOD
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#elif defined G2O_HAVE_CSPARSE
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#endif





using namespace Eigen;
using namespace std;




class Sample
{
public:
  static int uniform(int from, int to);
  static double uniform();
  static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr)
{
  return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma)
{
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to)
{
  return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform()
{
  return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma)
{
  return gauss_rand(0., sigma);
}



int main(int argc,char *argv[])
{
	ros::init(argc,argv,"sba_extractor");
 	typedef g2o::BlockSolver< g2o::BlockSolverTraits<-1, -1> >  SlamBlockSolver;
  	typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	std::cout<<SlamBlockSolver::PoseMatrixType<<std::endl;
	/*
	
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
	linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();


  	g2o::BlockSolver_6_3 * solver_ptr
      	= new g2o::BlockSolver_6_3(linearSolver);

  	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

  	optimizer.setAlgorithm(solver);
	
	std::cout<<"made g2o"<<std::endl;


	double f,cx,cy,T;
	f=801.99886;
	cx=505.37826;
	cy=383.36684;

	T=-96.352977;

	Eigen::Matrix<double,3,4> Pl,Pr;
	Pl.setZero();
	Pr.setZero();

	//init matrices


	Pl(2,2)=1;
	Pl(0,0)=f;
	Pl(1,1)=f;
	Pl(0,2)=cx;
	Pl(1,2)=cy;

	Pr=Pl;

	Pr(0,3)=T;



	std::cout<<Pl<<std::endl;
	std::cout<<Pr<<std::endl;


	// set pose vertices

	stereoVertex *frameA,*frameB;
	frameA=new stereoVertex();
	frameB=new stereoVertex();
	frameA->setId(1);
	frameA->setFixed(true);
	optimizer.addVertex(frameA);


	frameB->setId(2);
	optimizer.addVertex(frameB);
	

	frameA->Pl=Pl;
	frameA->Pr=Pr;

	frameB->Pl=Pl;
	frameB->Pr=Pr;



	int totalLandmarks=5;
	double guassianNoise=0.1;
	Eigen::Matrix<double,4,4> trueH;
	trueH.setIdentity();
	trueH(0,3)=-0.2;
	trueH(2,3)=-2;


	std::vector<Eigen::Vector3d> idealPoints;

	std::vector<Eigen::Vector3d> Aprojections,Bprojections;

	int VertexId=3;
	int edgeId=1;

	for(int ptIndex=0;ptIndex<totalLandmarks;ptIndex++)
	{
		std::cout<<VertexId<<","<<edgeId<<std::endl;
		Eigen::Vector4d currentPoint((Sample::uniform()-0.5)*3,
                                   Sample::uniform()-0.5,
                                   Sample::uniform()*2+10,1);

		Eigen::Vector4d shiftedPoint=trueH*currentPoint;	
		Eigen::Vector3d Al,Ar,Bl,Br;
		Eigen::Vector4d mVectA,mVectB;

		double arrayA[3],arrayB[3];

		g2o::VertexPointXYZ * v_p
        = new g2o::VertexPointXYZ();


		v_p->setId(VertexId);
		//v_p->setMarginalized(true);
		//v_p->setFixed(true);
		Al=Pl*(currentPoint);
		Al/=Al(2,0);
		Ar=Pr*currentPoint;
		Ar/=Ar(2,0);


		mVectA.block<2,1>(0,0)=Al.block<2,1>(0,0);
		mVectA.block<2,1>(2,0)=Ar.block<2,1>(0,0);

		arrayA[0]=currentPoint(0,0);
		arrayA[1]=currentPoint(1,0);
		arrayA[2]=currentPoint(2,0);
		v_p->setEstimateData(&arrayA[0]);
		optimizer.addVertex(v_p);


		landmarkEdge * e
			= new landmarkEdge();
		e->setMeasurement(mVectA);
		e->vertices()[0]=optimizer.vertex(1);
		e->vertices()[1]=optimizer.vertex(VertexId);
		edgeId++;


		optimizer.addEdge(e);
		///////////////////
		//frame B
		///////////////////


		Bl=Pl*shiftedPoint;
		Bl/=Bl(2,0);

		Br=Pr*shiftedPoint;
		Br/=Br(2,0);

		mVectB.block<2,1>(0,0)=Bl.block<2,1>(0,0);
		mVectB.block<2,1>(2,0)=Br.block<2,1>(0,0);

		g2o::VertexPointXYZ * v_B
			= new g2o::VertexPointXYZ();	

		arrayB[0]=shiftedPoint(0,0);
		arrayB[1]=shiftedPoint(1,0);
		arrayB[2]=shiftedPoint(2,0);
			
		v_B->setId(VertexId+1);
		//v_B->setMarginalized(true);
		//v_B->setFixed(true);
		v_B->setEstimateData(&arrayB[0]);

		optimizer.addVertex(v_B);




		landmarkEdge * eN
			= new landmarkEdge();
		eN->setMeasurement(mVectB);
		eN->vertices()[0]=optimizer.vertex(2);
		eN->vertices()[1]=optimizer.vertex(VertexId+1);
		edgeId++;

		VertexId+=2;

	}

	std::cout<<"setup\n";
	optimizer.setVerbose(true);
 	optimizer.initializeOptimization();


	optimizer.optimize(100);
	std::cout<<"FINISHED"<<std::endl;
	*/
	return 0;
}
 
