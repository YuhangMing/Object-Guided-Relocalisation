#ifndef GRPAH_OPRIMIZER_H
#define GRPAH_OPRIMIZER_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
// #include "g2o/types/icp/types_icp.h"

// #include "g2o/types/icp/g2o_types_icp_api.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/slam3d/types_slam3d.h"

namespace g2o{

class EdgeGenICP
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
    g2o::Vector3 pos0, pos1;
    g2o::Matrix3 cov0, cov1;
    EdgeGenICP()
    {
        pos0.setZero();
        pos1.setZero();
        cov0.setIdentity();
        cov1.setZero();
    }
};

class Edge_V_V_GenICP : public g2o::BaseBinaryEdge<3, EdgeGenICP, g2o::VertexSE3, g2o::VertexSE3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Edge_V_V_GenICP();
    Edge_V_V_GenICP(const Edge_V_V_GenICP* e);
    // Matrix3 cov0, cov1;

    // I/O functions
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    // return the error as a 3-vector
    virtual void computeError();

    // try analytic jacobians
// #ifdef GICP_ANALYTIC_JACOBIANS
    virtual void linearizeOplus();
// #endif

    // global derivative matrices
    static g2o::Matrix3 dRidx;
	static g2o::Matrix3 dRidy;
	static g2o::Matrix3 dRidz; // differential quat matrices
};

}

namespace fusion{

class Optimizer
{
public:
    void static GeneralICP(std::vector<Eigen::Vector3d> ref_pts, 
                           std::vector<Eigen::Vector3d> src_pts,
                           std::vector<Eigen::Matrix3d> ref_cov,
                           std::vector<Eigen::Matrix3d> src_cov,
                           Eigen::Matrix4d &pose);

};

}

#endif