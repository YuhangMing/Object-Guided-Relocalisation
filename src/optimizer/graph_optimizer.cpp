#include "g2o/core/factory.h"
#include "g2o/stuff/macros.h"
#include "optimizer/graph_optimizer.h"

// using namespace Eigen;

namespace g2o {

// namespace fusion{

g2o::Matrix3 Edge_V_V_GenICP::dRidx; // differential quat matrices
g2o::Matrix3 Edge_V_V_GenICP::dRidy; // differential quat matrices
g2o::Matrix3 Edge_V_V_GenICP::dRidz; // differential quat matrices

//
// Rigid 3D constraint between poses, given fixed point offsets
//
// input two matched points between the frames
// first point belongs to the first frame, position and cov
// second point belongs to the second frame, position and cov
// the measurement variable has type EdgeGenICP

Edge_V_V_GenICP::Edge_V_V_GenICP()
    : g2o::BaseBinaryEdge<3, EdgeGenICP, g2o::VertexSE3, g2o::VertexSE3>()
{
    dRidx << 0.0,0.0,0.0,
        0.0,0.0,2.0,
        0.0,-2.0,0.0;
    dRidy  << 0.0,0.0,-2.0,
        0.0,0.0,0.0,
        2.0,0.0,0.0;
    dRidz  << 0.0,2.0,0.0,
        -2.0,0.0,0.0,
        0.0,0.0,0.0;
}

// Copy constructor
Edge_V_V_GenICP::Edge_V_V_GenICP(const Edge_V_V_GenICP* e)
    : g2o::BaseBinaryEdge<3, EdgeGenICP, g2o::VertexSE3, g2o::VertexSE3>()
{
    _vertices[0] = const_cast<g2o::HyperGraph::Vertex*> (e->vertex(0));
    _vertices[1] = const_cast<g2o::HyperGraph::Vertex*> (e->vertex(1));

    _measurement.pos0 = e->measurement().pos0;
    _measurement.pos1 = e->measurement().pos1;
    _measurement.cov0 = e->measurement().cov0;
    _measurement.cov1 = e->measurement().cov1;

    // cov0 = e->cov0;
    // cov1 = e->cov1;
}

bool Edge_V_V_GenICP::read(std::istream& is){}
bool Edge_V_V_GenICP::write(std::ostream& os) const{}

void Edge_V_V_GenICP::computeError()
{
    // from <ViewPoint> to <Point>
    const g2o::VertexSE3 *vp0 = static_cast<const g2o::VertexSE3*>(_vertices[0]);
    const g2o::VertexSE3 *vp1 = static_cast<const g2o::VertexSE3*>(_vertices[1]);

    // get vp1 into vp0 frame
    g2o::Vector3 p1;
    p1 = vp1->estimate() * measurement().pos1;
    p1 = vp0->estimate().inverse() * p1;

    // euclidean distance
    _error = p1 - measurement().pos0;

    // update information matrix
    const g2o::Matrix3 transform = ( vp0->estimate().inverse() *  vp1->estimate() ).matrix().topLeftCorner<3,3>();
    information() = ( _measurement.cov0 + transform * _measurement.cov1 * transform.transpose() ).inverse();

}

// Jacobian
// [ -R0'*R1 | R0 * dRdx/ddx * 0p1 ]
// [  R0'*R1 | R0 * dR'dx/ddx * 0p1 ]

// #ifdef GICP_ANALYTIC_JACOBIANS
// jacobian defined as:
//    f(T0,T1) =  dR0.inv() * T0.inv() * (T1 * dR1 * p1 + dt1) - dt0
//    df/dx0 = [-I, d[dR0.inv()]/dq0 * T01 * p1]
//    df/dx1 = [R0, T01 * d[dR1]/dq1 * p1]
void Edge_V_V_GenICP::linearizeOplus()
{
    g2o::VertexSE3* vp0 = static_cast<g2o::VertexSE3*>(_vertices[0]);
    g2o::VertexSE3* vp1 = static_cast<g2o::VertexSE3*>(_vertices[1]);

    // topLeftCorner<3,3>() is the rotation matrix
    g2o::Matrix3 R0T = vp0->estimate().matrix().topLeftCorner<3,3>().transpose();
    g2o::Vector3 p1 = measurement().pos1;

    // this could be more efficient
    if (!vp0->fixed())
      {
        g2o::Isometry3 T01 = vp0->estimate().inverse() * vp1->estimate();
        g2o::Vector3 p1t = T01 * p1;
        _jacobianOplusXi.block<3,3>(0,0) = -g2o::Matrix3::Identity();
        _jacobianOplusXi.block<3,1>(0,3) = dRidx*p1t;
        _jacobianOplusXi.block<3,1>(0,4) = dRidy*p1t;
        _jacobianOplusXi.block<3,1>(0,5) = dRidz*p1t;
      }

    if (!vp1->fixed())
      {
        g2o::Matrix3 R1 = vp1->estimate().matrix().topLeftCorner<3,3>();
        R0T = R0T*R1;
        _jacobianOplusXj.block<3,3>(0,0) = R0T;
        _jacobianOplusXj.block<3,1>(0,3) = R0T*dRidx.transpose()*p1;
        _jacobianOplusXj.block<3,1>(0,4) = R0T*dRidy.transpose()*p1;
        _jacobianOplusXj.block<3,1>(0,5) = R0T*dRidz.transpose()*p1;
      }
}
// #endif

}

namespace fusion{

void Optimizer::GeneralICP(std::vector<Eigen::Vector3d> ref_pts, 
                           std::vector<Eigen::Vector3d> src_pts,
                           std::vector<Eigen::Matrix3d> ref_cov,
                           std::vector<Eigen::Matrix3d> src_cov,
                           Eigen::Matrix4d &pose)
{
    // initialize optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    // set solver
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>())
    );
    optimizer.setAlgorithm(solver);

    // set reference pose node
    Eigen::Vector3d t(0, 0, 0);
    Eigen::Quaterniond q;
    q.setIdentity();
    Eigen::Isometry3d cam;
    cam = q;
    cam.translation() = t;
    g2o::VertexSE3 *vc0 = new g2o::VertexSE3();
    vc0->setEstimate(cam);
    vc0->setId(0);
    vc0->setFixed(true);
    optimizer.addVertex(vc0);

    // set current pose node
    Eigen::Matrix3d init_rot = pose.topLeftCorner(3,3);
    Eigen::Vector3d init_t = pose.topRightCorner(3,1);
    Eigen::Quaterniond init_q(init_rot);
    Eigen::Isometry3d init_cam;
    init_cam = init_q;
    init_cam.translation() = init_t;
    g2o::VertexSE3 *vc1 = new g2o::VertexSE3();
    vc1->setEstimate(init_cam);
    vc1->setId(1);
    optimizer.addVertex(vc1);

    // set up GICP edge
    assert(ref_pts.size() == src_pts.size());
    for(size_t i=0; i<ref_pts.size(); ++i)
    {
        g2o::Edge_V_V_GenICP *e = new g2o::Edge_V_V_GenICP();
        e->setVertex(0, vc0);
        e->setVertex(1, vc1);

        // e->cov0 = ref_cov[i];
        // e->cov1 = Eigen::Matrix3d::Identity();

        g2o::EdgeGenICP meas;
        meas.pos0 = ref_pts[i];
        meas.pos1 = src_pts[i];
        // meas.cov0 = ref_cov[i];
        // meas.cov1 = src_cov[i];
        
        e->setMeasurement(meas);
        e->information() = ref_cov[i].inverse();
        // e->information().setIdentity();
        // std::cout << e->information() << std::endl;
        optimizer.addEdge(e);
    }
    std::cout << "[ In G2O graph: "
              << "nVertices=" << optimizer.vertices().size()
              << "; nEdges=" << optimizer.edges().size() 
              << ". ]" << std::endl;

    // start optimization
    optimizer.initializeOptimization();
    // optimizer.computeActiveErrors();
    // std::cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << std::endl;

    optimizer.setVerbose(false);
    optimizer.optimize(5);

    // recover optimized pose
    pose = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second)
            ->estimate().matrix();

}

}