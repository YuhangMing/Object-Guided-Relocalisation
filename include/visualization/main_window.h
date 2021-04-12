#ifndef GL_WINDOW_H
#define GL_WINDOW_H

#include "system.h"
#include "utils/settings.h"
// #include "visualization/drawer.h"
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

class MainWindow
{
public:
    ~MainWindow();
    MainWindow(const char *name = "Untitled", size_t width = 640, size_t height = 480, bool bDisplay=true);

    //! Do not copy this class
    MainWindow(const MainWindow &) = delete;
    MainWindow &operator=(const MainWindow &) = delete;

    //! Main loop
    void Render();

    void ResetAllFlags();
    void SetVertexSize(size_t Size);
    void SetRGBSource(cv::Mat RgbImage);
    void SetDepthSource(cv::Mat DepthImage);
    void SetDetectedSource(cv::Mat DetectedImage);
    void SetCurrentCamera(Eigen::Matrix4f T);
    void SetSystem(fusion::System *sys);
    void SetPause();

    // void SetRenderScene(cv::Mat SceneImage);
    // void SetFeatureImage(cv::Mat featureImage);
    // void SetNOCSMap(cv::Mat SceneImage);
    // void SetMask(cv::Mat mask);

    bool IsPaused();
    bool mbFlagRestart;
    bool mbFlagUpdateMesh;

    // bool bDisplayOtherPoses;

    /* Old mesh drawing functions
    size_t VERTEX_COUNT;
    size_t MAX_VERTEX_COUNT;

    float *GetMappedVertexBuffer();
    float *GetMappedNormalBuffer();
    unsigned char *GetMappedColourBuffer();
    */

private:
    //! Window Title
    std::string WindowName;

    void SetupDisplays();
    void SetupGLFlags();
    void InitTextures();
    void InitGlSlPrograms();
    void RegisterKeyCallback();

    //! Displayed Views
    pangolin::View *mpViewSideBar;
    pangolin::View *mpViewScene;
    pangolin::View *mpViewMesh;
    pangolin::View *mpViewMenu;
    pangolin::View *mpViewRGB;
    pangolin::View *mpViewRelocView;
    // Semantic & Reloc diasbled for now
    // pangolin::View *mpViewDepth;
    // pangolin::View *mpViewNOCSMap;
    // pangolin::View *mpViewMask;

    //! Displayed textures
    pangolin::GlTexture TextureRGB;
    pangolin::GlTexture TextureDepth;
    pangolin::GlTexture TextureDetected; // used as backup rgb display
    // pangolin::GlTexture TextureScene;
    // pangolin::GlTexture TextureNOCSMap;
    // pangolin::GlTexture TextureMask;

    //! Main 3D View Camera
    std::shared_ptr<pangolin::OpenGlRenderState> CameraView;

    //! GUI buttons and checkboxes
    // int nIndicator, cIndicator;
    std::shared_ptr<pangolin::Var<bool>> BtnReset;
    std::shared_ptr<pangolin::Var<bool>> BtnSaveMap;
    std::shared_ptr<pangolin::Var<bool>> BtnSetLost;
    std::shared_ptr<pangolin::Var<bool>> BtnReadMap;
    std::shared_ptr<pangolin::Var<bool>> BoxPaused;
    std::shared_ptr<pangolin::Var<bool>> BoxPureReloc;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayImage;
    // std::shared_ptr<pangolin::Var<bool>> BoxDisplayDepth;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayScene;
    // std::shared_ptr<pangolin::Var<int>> BarSwitchMap;
    std::shared_ptr<pangolin::Var<int>> BarSwitchSubmap;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayCamera;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayKeyCameras;
    std::shared_ptr<pangolin::Var<int>> BarSwitchCuboid;
    std::shared_ptr<pangolin::Var<bool>> BoxPrimaryCuboid;
    /* Semantic & Reloc diasbled for now
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayDetected;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayRelocTrajectory;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayKeyPoint;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayCuboids;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayPtCloud;
    std::shared_ptr<pangolin::Var<int>> BarSwitchCuboidsReloc;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayPtCloudReloc;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayWorldOrigin;
    // std::shared_ptr<pangolin::Var<bool>> BoxRecordData;
    std::shared_ptr<pangolin::Var<int>> BarSwitchObject;
    // std::shared_ptr<pangolin::Var<bool>> BoxDisplayCuboidsAVG;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayMainCuboids;
    */

    //! New Draw Mesh Function
    void DrawMesh(int idx);
    void DeleteMesh();

    //! Draw Cuboid Function
    bool bUseGT = false;
    void DrawCuboids(std::vector<std::pair<int, std::vector<float>>> label_dim_pairs, 
                     int idx, bool bRel, int usePose=0);
    void DrawPtClouds(int num_objs, int idx, bool bRel);
    // TEST //
    void DrawTestCuboids(int label);

    //! GL Shading program
    pangolin::GlSlProgram ShadingProg, ShadingColorProg;

    //! Current Camera Pose
    Eigen::Matrix4f CameraPose;
    
    //! system ref
    fusion::System *slam;

    // //! Key Frame Poses
    // std::vector<Eigen::Matrix4f> ListOfKeyCameras;


    //! color palette
    /*
        bg - black
        bottle - orange
        bowl - olive 
        camera - purple
        can - yellow 
        laptop - navy
        mug - cyan
    */
    float palette[7][3] = {
        {0.,        0.,        0.},
        {1.,        165./255., 0.},
        {128./255., 128./255., 0},
        {128./255., 0.,        128./255.},
        {1.,        1.,        0},
        {0.,        0.,        1.},
        {0.,        1.,        1.}
    };

    // //! key point array
    // float *keypoints;
    // size_t sizeKeyPoint;
    // size_t maxSizeKeyPoint;

    /* Old mesh drawing functions
    void InitMeshBuffers();

    //! Acquire Mehs Functions
    void UpdateMeshWithNormal();
    void UpdateMeshWithColour();
    
    //! Draw Mesh Functions
    void DrawMeshShaded();
    void DrawMeshColoured();
    void DrawMeshNormalMapped();
    
    //! Mesh Vertices
    // following variables store teh vertex, normal and color info from the map
    pangolin::GlBufferCudaPtr BufferVertex;
    pangolin::GlBufferCudaPtr BufferNormal;
    pangolin::GlBufferCudaPtr BufferColour;

    //! Registered CUDA Ptrs
    std::shared_ptr<pangolin::CudaScopedMappedPtr> MappedVertex;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> MappedNormal;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> MappedColour;

    //! Vertex Array Objects
    //! Cannot find a replacement in Pangolin
    GLuint VAOShade, VAOColour;
    // stores vertex buffer
    */

};

#endif