#ifndef SUBMAP_MANAGER_H
#define SUBMAP_MANAGER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <mutex>
#include "system.h"
#include "tracking/rgbd_frame.h"
#include "mapping/VoxelMap.h"
#include "mapping/RayTraceEngine.h"
#include "mapping/MeshEngine.h"
#include "mapping/ObjectMap.h"

namespace fusion
{

// class DenseMapping;
class DenseOdometry;

class SubmapManager {

public:
	SubmapManager();
	~SubmapManager();

	void Create(int submapIdx, bool bTrack=false, bool bRender=false);
	void AddKeyFrame(Eigen::Matrix4f kfPose);
	void ResetSubmaps();

	std::vector<MapStruct *> getDenseMaps();
	std::vector<Eigen::Matrix<float, 4, 4>> GetKFPoses();
	std::vector<std::pair<int, std::vector<float>>> GetObjects(bool bMain);
	int GetNumObjs() const;
	
	void writeMapToDisk();
	void readMapFromDisk();
	void writePosesToText(std::string file_name, std::vector<Eigen::Matrix4d> vPoses);
	void readPosesFromText(std::string file_name, std::vector<Eigen::Matrix4d>& vPoses);
	
	//!! Remove vPoses after orthogonal issue in pose loading
	std::vector<Eigen::Matrix4d> vSubmapPoses;

	MeshEngine *pMesher;
	RayTraceEngine *pRayTracer;
	std::vector< MapStruct * > vActiveSubmaps;
	std::vector< ObjectMap * > vObjectMaps;

	

	/*
	void Create(int submapIdx, RgbdImagePtr ref_img, bool bTrack=false, bool bRender=true);
	
	float CheckVisPercent(int submapIdx);
	void CheckActive();
	void CheckTrackAndRender(int cur_frame_id, int max_perct_idx);

	void estimate_world_plane(RgbdFramePtr frame);
	void GetPoints(float *pt3d, size_t &count, size_t max_size);
	std::vector<std::pair<int, std::vector<float>>> GetObjectCuboids();
	std::vector<float> GetObjectCentroidAxes(int idx_obj);
	int GetObjectPts(float *points, size_t &count, int idx_obj);
	int GetNumObjs() const;
	int GetObjectPointCloud(float *points, size_t &count, int idx_obj);
	std::vector<float> GetPlaneNormals();

	void SetTracker(std::shared_ptr<DenseOdometry> pOdometry);
	void SetExtractor(std::shared_ptr<FeatureExtractor> pExtractor);

	// pure relocalization
	// v of semantic maps, each map contains a vector of cuboids
	std::vector< std::vector<std::shared_ptr<Cuboid3d>> >  vSMaps;
	void readSMapFromDisk(std::string file_name);
	*/

	std::shared_ptr<DenseOdometry> odometry;

	bool bKFCreated;
	bool bHasNewSM;
	int renderIdx;	// for active submaps, index of the submapp is constructing

	// submap storage
	// std::vector< std::shared_ptr<DenseMapping> > all_submaps;	// discard this variable, all_submaps = active + passive
	// std::vector< std::shared_ptr<DenseMapping> > active_submaps;
	// std::vector< std::shared_ptr<DenseMapping> > passive_submaps;
	// std::vector< int > activeTOpassiveIdx;

	// std::map< int, std::vector<Eigen::Vector4f> > mPassiveMPs;
	// std::map< int, std::vector<Eigen::Vector4f> > mPassiveNs;

protected:

private:
	std::mutex mMutexDownload;
	int ref_frame_id;
};


} // namespace fusion

#endif