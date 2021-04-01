#include <vector>
#include <Eigen/Core>
#include "tracking/rgbd_frame.h"
#include "detection/map_object.h"

namespace fusion
{

class ObjectMap
{
public:
    ObjectMap(int id);
    ~ObjectMap();

    int denseMapId;
    std::vector<Eigen::Matrix4f> vKFs;
    
    std::vector<std::shared_ptr<Object3d>> v_objects;                         // primary object, used for relocalisation
    std::map<int, std::vector<std::shared_ptr<Object3d>>> object_dictionary;  // back-up dictionary, mainly used in map construction
    void update_objects(RgbdFramePtr frame);

    void writeObjectsToDisk(std::string file_name);
    void readObjectsFromDisk(std::string file_name);

    // // plane
    // std::vector<Eigen::Vector3f> plane_normals;
    // void update_planes(RgbdFramePtr frame);
    
    // // cuboid
    // std::vector<std::shared_ptr<Cuboid3d>> object_cuboids;  // primary objects
    // std::map<int, std::vector<std::shared_ptr<Cuboid3d>>> cuboid_dictionary;  // back-up dictionary
    // void update_cuboids(RgbdFramePtr frame);
    // void estimate_cuboids(RgbdImagePtr frame);                      // old-version GPU
    // void estimate_cuboids(RgbdFramePtr frame, bool tracking_lost);  // old-version CPU
    // // quadric
    // std::vector<std::shared_ptr<Quadric3d>> object_quadrics;
    // void estimate_quadrics(RgbdFramePtr frame, bool tracking_lost);

private:

};

} // namespace fusion