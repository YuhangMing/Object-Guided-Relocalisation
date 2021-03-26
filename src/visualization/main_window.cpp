#include "visualization/main_window.h"

#define ENTER_KEY 13

MainWindow::MainWindow(const char *name, size_t width, size_t height, bool bDisplay)
    : mbFlagRestart(false), WindowName(name)
    // , mbFlagUpdateMesh(false), VERTEX_COUNT(0), MAX_VERTEX_COUNT(20000000)
    // , sizeKeyPoint(0),maxSizeKeyPoint(8000000)
{
    ResetAllFlags();

    pangolin::CreateWindowAndBind(WindowName, width, height);
    // keypoints = (float *)malloc(sizeof(float) * maxSizeKeyPoint);

    SetupGLFlags();
    SetupDisplays();
    RegisterKeyCallback();
    InitTextures();
    // InitMeshBuffers();
    InitGlSlPrograms();

    // cIndicator = 1;
    // nIndicator = 0;

    // bDisplayOtherPoses = bDisplay;
    // bRecording = false;
}

MainWindow::~MainWindow()
{
    // delete keypoints;
    pangolin::DestroyWindow(WindowName);
    std::cout << "opengl released. " << std::endl;
}


void MainWindow::SetupGLFlags()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void MainWindow::SetupDisplays()
{
    CameraView = std::make_shared<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    auto MenuDividerLeft = pangolin::Attach::Pix(200);
    float RightSideBarDividerLeft = 0.7f;

    pangolin::CreatePanel("Menu").SetBounds(0, 1, 0, MenuDividerLeft);

    // name of the button, default value, shi fou you xuan ze kuang
    BtnReset = std::make_shared<pangolin::Var<bool>>("Menu.RESET", false, false);
    BtnSetLost = std::make_shared<pangolin::Var<bool>>("Menu.Set Lost", false, false);
    BtnSaveMap = std::make_shared<pangolin::Var<bool>>("Menu.Save Map", false, false);
    BtnReadMap = std::make_shared<pangolin::Var<bool>>("Menu.Read Map", false, false);
    BoxPaused = std::make_shared<pangolin::Var<bool>>("Menu.PAUSE", true, true);
    BoxDisplayImage = std::make_shared<pangolin::Var<bool>>("Menu.KeyFrame Image", true, true);
    BoxDisplayScene = std::make_shared<pangolin::Var<bool>>("Menu.Rendered Scene", true, true);
    // BoxDisplayDepth = std::make_shared<pangolin::Var<bool>>("Menu.Live Depth", true, true);
    // BoxDisplayDetected = std::make_shared<pangolin::Var<bool>>("Menu.Detected Result", true, true);
    
    // display name, current val, min, max;
    // BarSwitchMap = std::make_shared<pangolin::Var<int>>("Menu.Display Map", 1, 0, 2);
    BoxDisplayCamera = std::make_shared<pangolin::Var<bool>>("Menu.Current Camera", true, true);
    
    /* Semantic disabled for now
    BoxDisplayKeyCameras = std::make_shared<pangolin::Var<bool>>("Menu.Stored KeyFrames", false, true);
    BoxDisplayRelocTrajectory = std::make_shared<pangolin::Var<bool>>("Menu.Reloc Frames", false, true);
    
    BoxDisplayKeyPoint = std::make_shared<pangolin::Var<bool>>("Menu.Stored KeyPoint", false, true);
    BarSwitchCuboid = std::make_shared<pangolin::Var<int>>("Menu.Display Cuboid", 7, 0, 7);
    BoxDisplayCuboids = std::make_shared<pangolin::Var<bool>>("Menu.Object Main Cuboids", true, true);
    BoxDisplayAllCuboids = std::make_shared<pangolin::Var<bool>>("Menu.Object All Cuboids", false, true);
    BoxDisplayPtCloud = std::make_shared<pangolin::Var<bool>>("Menu.Object PtCloud", false, true);
    BarSwitchCuboidsReloc = std::make_shared<pangolin::Var<int>>("Menu.Frame Cuboids", 0, 0, 2);
    // BoxDisplayCuboidsReloc = std::make_shared<pangolin::Var<bool>>("Menu.Frame Cuboids", false, true);
    BoxDisplayPtCloudReloc = std::make_shared<pangolin::Var<bool>>("Menu.Frame PtCloud", false, true);
    BoxDisplayWorldOrigin = std::make_shared<pangolin::Var<bool>>("Menu.Wrold Origin", false, true);
    // BoxRecordData = std::make_shared<pangolin::Var<bool>>("Menu.Record Dataset", true, true);
    */

    mpViewSideBar = &pangolin::Display("Right Side Bar");
    mpViewSideBar->SetBounds(0, 1, RightSideBarDividerLeft, 1);
    mpViewRGB = &pangolin::Display("RGB");
    mpViewRGB->SetBounds(0.5, 1, 0, 1);
    mpViewRelocView = &pangolin::Display("Reloc View");
    mpViewRelocView->SetBounds(0, 0.5, 0, 1);
    
    // mpViewRelocView = &pangolin::Display("Reloc View");
    // mpViewRelocView->SetBounds(0.25, 0.5, 0, 0.5);
    // mpViewDepth = &pangolin::Display("Depth");
    // mpViewDepth->SetBounds(0.25, 0.5, 0.5, 1);
    
    // mpViewNOCSMap = &pangolin::Display("NOCS Map");
    // mpViewNOCSMap->SetBounds(0, 0.25, 0.5, 1);
    // mpViewMask = &pangolin::Display("Seg Mask");
    // mpViewMask->SetBounds(0, 0.25, 0, 0.5);

    mpViewScene = &pangolin::Display("Scene");
    mpViewScene->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft);
    mpViewMesh = &pangolin::Display("Mesh");
    mpViewMesh->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft).SetHandler(new pangolin::Handler3D(*CameraView));

    mpViewSideBar->AddDisplay(*mpViewRGB);
    mpViewSideBar->AddDisplay(*mpViewRelocView);
    // mpViewSideBar->AddDisplay(*mpViewDepth);
    // mpViewSideBar->AddDisplay(*mpViewNOCSMap);
    // mpViewSideBar->AddDisplay(*mpViewMask);
}

void MainWindow::RegisterKeyCallback()
{
    //! Retart the system
    pangolin::RegisterKeyPressCallback('r', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    pangolin::RegisterKeyPressCallback('R', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    //! Pause / Resume the system
    pangolin::RegisterKeyPressCallback(ENTER_KEY, pangolin::ToggleVarFunctor("Menu.PAUSE"));
    //! Display keyframes
    pangolin::RegisterKeyPressCallback('c', pangolin::ToggleVarFunctor("Menu.Display KeyFrame"));
    pangolin::RegisterKeyPressCallback('C', pangolin::ToggleVarFunctor("Menu.Display KeyFrame"));
    //! Save Maps
    pangolin::RegisterKeyPressCallback('s', pangolin::SetVarFunctor<bool>("Menu.Save Map", true));
    pangolin::RegisterKeyPressCallback('S', pangolin::SetVarFunctor<bool>("Menu.Save Map", true));
    //! Load Maps
    pangolin::RegisterKeyPressCallback('l', pangolin::SetVarFunctor<bool>("Menu.Read Map", true));
    pangolin::RegisterKeyPressCallback('L', pangolin::SetVarFunctor<bool>("Menu.Read Map", true));
}

void MainWindow::InitTextures()
{
    TextureRGB.Reinitialise(
        640, 480,
        GL_RGB,
        true,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);

    TextureDepth.Reinitialise(
        640, 480,
        GL_LUMINANCE,
        true,
        0,
        GL_LUMINANCE,
        GL_UNSIGNED_BYTE,
        NULL);

    /* Semantic & Reloc diasbled for now
    TextureDetected.Reinitialise(
        640, 480,
        GL_RGB,
        true,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);

    TextureScene.Reinitialise(
        640, 480,
        GL_RGBA,
        true,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        NULL);

    TextureNOCSMap.Reinitialise(
        640, 480,
        GL_RGB,
        true,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);

    TextureMask.Reinitialise(
        640, 480,
        GL_LUMINANCE,
        true,
        0,
        GL_LUMINANCE,
        GL_UNSIGNED_BYTE,
        NULL);
    */
}

void MainWindow::InitMeshBuffers()
{
    // auto size = sizeof(float) * 3 * MAX_VERTEX_COUNT;

    // BufferVertex.Reinitialise(
    //     pangolin::GlArrayBuffer,
    //     size,
    //     cudaGLMapFlagsWriteDiscard,
    //     GL_STATIC_DRAW);

    // BufferNormal.Reinitialise(
    //     pangolin::GlArrayBuffer,
    //     size,
    //     cudaGLMapFlagsWriteDiscard,
    //     GL_STATIC_DRAW);

    // BufferColour.Reinitialise(
    //     pangolin::GlArrayBuffer,
    //     size,
    //     cudaGLMapFlagsWriteDiscard,
    //     GL_STATIC_DRAW);

    // MappedVertex = std::make_shared<pangolin::CudaScopedMappedPtr>(BufferVertex);
    // MappedNormal = std::make_shared<pangolin::CudaScopedMappedPtr>(BufferNormal);
    // MappedColour = std::make_shared<pangolin::CudaScopedMappedPtr>(BufferColour);

    // glGenVertexArrays(1, &VAOShade);
    // glBindVertexArray(VAOShade);

    // BufferVertex.Bind();
    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // glEnableVertexAttribArray(0);

    // BufferNormal.Bind();
    // glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // glEnableVertexAttribArray(1);

    // glGenVertexArrays(1, &VAOColour);
    // glBindVertexArray(VAOColour);

    // BufferVertex.Bind();
    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // glEnableVertexAttribArray(0);

    // BufferColour.Bind();
    // glVertexAttribPointer(2, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    // glEnableVertexAttribArray(2);

    // // previous buffers are unbinded automatically when binding next buffer.
    // // so only last one need to call Unbind explicitly
    // BufferColour.Unbind();
    // glBindVertexArray(0);
}

void MainWindow::InitGlSlPrograms()
{
    const char vertexShader[] =
        "#version 330\n"
        "\n"
        "layout(location = 0) in vec3 position;\n"
        "layout(location = 1) in vec3 a_normal;\n"
        "uniform mat4 mvpMat;\n"
        "uniform mat4 Tmw;\n"
        "uniform float colourTaint;\n"
        "out vec3 shaded_colour;\n"
        "\n"
        "void main(void) {\n"
        "    gl_Position = mvpMat * Tmw * vec4(position, 1.0);\n"
        "    vec3 lightpos = vec3(5, 5, 5);\n"
        "    const float ka = 0.3;\n"
        "    const float kd = 0.5;\n"
        "    const float ks = 0.2;\n"
        "    const float n = 20.0;\n"
        "    float ax = 1.0;\n"
        "    float dx = 1.0;\n"
        "    float sx = 1.0;\n"
        "    const float lx = 1.0;\n"
        "    vec3 L = normalize(lightpos - position);\n"
        "    vec3 V = normalize(vec3(0.0) - position);\n"
        "    vec3 R = normalize(2 * a_normal * dot(a_normal, L) - L);\n"
        "    float i1 = ax * ka * dx;\n"
        "    float i2 = lx * kd * dx * max(0.0, dot(a_normal, L));\n"
        "    float i3 = lx * ks * sx * pow(max(0.0, dot(R, V)), n);\n"
        "    float Ix = max(0.0, min(255.0, i1 + i2 + i3));\n"
        "    shaded_colour = vec3(Ix, Ix, Ix);\n"
        "}\n";

    const char fragShader[] =
        "#version 330\n"
        "\n"
        "in vec3 shaded_colour;\n"
        "out vec4 colour_out;\n"
        "void main(void) {\n"
        "    colour_out = vec4(shaded_colour, 1);\n"
        "}\n";
    ShadingProg.AddShader(pangolin::GlSlVertexShader, vertexShader);
    ShadingProg.AddShader(pangolin::GlSlFragmentShader, fragShader);
    // ShadingProg.AddShaderFromFile(
    //     pangolin::GlSlShaderType::GlSlVertexShader,
    //     "./glsl_shader/phong.vert");
    // ShadingProg.AddShaderFromFile(
    //     pangolin::GlSlShaderType::GlSlFragmentShader,
    //     "./glsl_shader/direct_output.frag");

    ShadingProg.Link();

    // color not used here.
    ShadingColorProg.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlVertexShader,
        "./glsl_shader/colour.vert");
    ShadingColorProg.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlFragmentShader,
        "./glsl_shader/direct_output.frag");
    ShadingColorProg.Link();
}

bool MainWindow::IsPaused()
{
    return *BoxPaused;
}

void MainWindow::SetPause()
{
    // pangolin::ToggleVarFunctor("Menu.PAUSE");
    *BoxPaused = true;
}

void MainWindow::ResetAllFlags()
{
    mbFlagRestart = false;
    // mbFlagUpdateMesh = false;
}

void MainWindow::SetRGBSource(cv::Mat RgbImage)
{
    TextureRGB.Upload(RgbImage.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void MainWindow::SetDepthSource(cv::Mat DepthImage)
{
    cv::Mat Depth_8bit = DepthImage.clone();
    double min, max;
    cv::minMaxLoc(Depth_8bit, &min, &max);
    Depth_8bit = Depth_8bit / max * 255;
    Depth_8bit.convertTo(Depth_8bit, CV_8U);
    TextureDepth.Upload(Depth_8bit.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
}

/* Semantic disabled for now
void MainWindow::SetDetectedSource(cv::Mat DetectedImage)
{
    TextureDetected.Upload(DetectedImage.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void MainWindow::SetRenderScene(cv::Mat SceneImage)
{
    TextureScene.Upload(SceneImage.data, GL_RGBA, GL_UNSIGNED_BYTE);
}

void MainWindow::SetNOCSMap(cv::Mat SceneImage)
{
    TextureNOCSMap.Upload(SceneImage.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void MainWindow::SetMask(cv::Mat mask)
{
    if(!mask.empty())
        TextureMask.Upload(mask.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
}

void MainWindow::SetFeatureImage(cv::Mat featureImage)
{
}
*/


void MainWindow::Render()
{
    auto t1 = std::chrono::system_clock::now();
    
    ResetAllFlags();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.f, 0.f, 0.f, 1.f);

    if (pangolin::Pushed(*BtnReset))
    {
        slam->restart();
        if (IsPaused())
            UpdateMeshWithNormal();
    }

    if (pangolin::Pushed(*BtnSetLost))
    {
        slam->setLost(true);
    }

    if (pangolin::Pushed(*BtnSaveMap))
    {
        slam->writeMapToDisk();
    }

    if (pangolin::Pushed(*BtnReadMap))
    {
        slam->readMapFromDisk();
        if (IsPaused())
            DrawMesh();
    }

    // if(bRecording)
    // {
        if (*BoxDisplayScene)
        {
            mpViewRelocView->Activate();
            TextureDepth.RenderToViewportFlipY();
        }
        if (*BoxDisplayImage)
        {
            mpViewRGB->Activate();
            TextureRGB.RenderToViewportFlipY();
        }
    // } 
    // else 
    // {
    //     if (*BoxDisplayScene)
    //     {
    //         mpViewRelocView->Activate();
    //         TextureScene.RenderToViewportFlipY();
    //     }
    //     if (*BoxDisplayImage)
    //     {
    //         mpViewRGB->Activate();
    //         TextureDetected.RenderToViewportFlipY();
    //     }
    // }
    
    // if (*BoxDisplayDepth)
    // {
    //     mpViewDepth->Activate();
    //     TextureDepth.RenderToViewportFlipY();
    // }
    // if (*BoxDisplayDetected){
    //     mpViewMask->Activate();
    //     TextureMask.RenderToViewportFlipY();
    //     mpViewNOCSMap->Activate();
    //     TextureNOCSMap.RenderToViewportFlipY();
    // }

    // -- Display 3D map as constructing ----
    // -- Could jeopardize the real-time performance
    mpViewMesh->Activate(*CameraView);
    if (!IsPaused())
        DeleteMesh();
    DrawMesh();
    // switch (*BarSwitchMap)
    // {
    //     case 2:
    //         // if(cIndicator == 0)
    //         // {
    //         //     UpdateMeshWithColour();
    //         //     cIndicator++;
    //         //     nIndicator = 0;
    //         // }
    //         // DrawMeshColoured();
    //         break;
    //     case 1:
    //         DrawMesh();
    //         // if(nIndicator == 0)
    //         // {
    //         //     UpdateMeshWithNormal();
    //         //     nIndicator++;
    //         //     cIndicator = 0;
    //         // }
    //         // DrawMeshShaded();
    //     default:
    //         // cIndicator = 0;
    //         // nIndicator = 0;
    //         break;
    // }
    // std::cout << "    DONE." << std::endl;
    Eigen::Matrix3f K;
    K << 580, 0, 320, 0, 580, 240, 0, 0, 1;
    if (*BoxDisplayCamera)
    {
        // Current Camera
        glColor4f(1.f, 0.f, 0.f, 1.f);
        pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, CameraPose, 0.04f);
        pangolin::glDrawAxis(CameraPose, 0.02f);
        glColor4f(1.f, 1.f, 1.f, 1.f);
        
        /* Reloc Disabled for now
        int reloc_id = slam->reloc_frame_id-1;
        if(reloc_id >= 0)
        {
            // GT for Relocalized Frames
            std::vector<Eigen::Matrix4f> gt_poses = slam->getGTposes();
            if(reloc_id < gt_poses.size()){
                glColor4f(0.f, 1.f, 0.f, 1.f);
                Eigen::Matrix4f pose = gt_poses[reloc_id];
                pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, pose, 0.04f);
                pangolin::glDrawAxis(pose, 0.02f);
                glColor4f(1.f, 1.f, 1.f, 1.f);
                if(bDisplayOtherPoses){
                    // NOCS Pose based relocalization result
                    // Current stores ORB baseline reloc result
                    std::vector<Eigen::Matrix4f> nocs_pose_results = slam->getNOCSPoseResults();
                    if(reloc_id < nocs_pose_results.size()){
                        glColor4f(0.f, 0.f, 1.f, 1.f);
                        auto nocs_pose = nocs_pose_results[reloc_id];
                        pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, nocs_pose, 0.02f);
                        pangolin::glDrawAxis(nocs_pose, 0.02f);
                        glColor4f(1.f, 1.f, 1.f, 1.f);
                    }
                }
            }
        }
        */
    }

    /* Semantic and Reloc Disabled for now
    if (*BoxDisplayKeyCameras)
    {
        auto keyframe_poses = slam->getKeyFramePoses();
        std::vector<Eigen::Matrix<float, 3, 1>> camera_centers;
        for (const auto &pose : keyframe_poses)
        {
            camera_centers.push_back(pose.topRightCorner(3, 1));
            pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, pose, 0.02f);
            pangolin::glDrawAxis(pose, 0.005f);
        }
        pangolin::glDrawVertices(camera_centers, GL_LINE_STRIP);
    }
    if (*BoxDisplayRelocTrajectory)
    {
        // if(reloc_poses.size() > 0)
        // {
        //     std::vector<Eigen::Matrix<float, 3, 1>> camera_centers;
        //     glColor4f(1.f, 0.f, 0.f, 1.f);
        //     for (const auto &pose : reloc_poses )
        //     {
        //         Eigen::Matrix<float, 3, 1> one_center = pose.topRightCorner(3, 1);
        //         pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, pose, 0.01f);
        //         // pangolin::glDrawAxis(pose, 0.005f);
        //     }
        //     // pangolin::glDrawVertices(camera_centers, GL_LINE_STRIP);
        //     // glColor4f(1.f, 1.f, 1.f, 1.f);
        // }

        auto reloc_poses_GT = slam->getRelocPosesGT();
        // if(reloc_poses_GT.size() > 0)
        // {
        //     std::vector<Eigen::Matrix<float, 3, 1>> camera_centers;
        //     glColor4f(0.f, 1.f, 0.f, 1.f);
        //     for(size_t sid=0; sid<3; ++sid)
        //     {
        //         size_t lower = sid*100;
        //         size_t upper = sid*100 + 100;
        //         for(size_t rfid=lower; rfid<upper; rfid++)
        //         {
        //             Eigen::Matrix<float, 3, 1> one_center = reloc_poses_GT[rfid].topRightCorner(3, 1);
        //             camera_centers.push_back(one_center);
        //         }
        //         pangolin::glDrawVertices(camera_centers, GL_LINE_STRIP);
        //         camera_centers.clear();
        //     }
        // }

        auto reloc_poses = slam->getRelocPoses();

        if(reloc_poses.size() > 0 && reloc_poses_GT.size() > 0)
        {
            std::vector<Eigen::Matrix<float, 3, 1>> gt_centers;
            std::vector<Eigen::Matrix<float, 3, 1>> est_centers;
            size_t last_draw_id = -10;
            Eigen::Matrix<float, 3, 1> last_draw_pos;
            last_draw_pos << 0, 0, 0;
            for(size_t sid=0; sid<3; ++sid)
            {
            
                size_t lower = sid*100;
                size_t upper = sid*100 + 100;
                for(size_t rfid=lower; rfid<upper; rfid++)
                {
                    Eigen::Matrix<float, 3, 1> one_gt_center = reloc_poses_GT[rfid].topRightCorner(3, 1);
                    gt_centers.push_back(one_gt_center);

                    Eigen::Matrix<float, 3, 1> one_est_center = reloc_poses[rfid].topRightCorner(3, 1);
                    if((one_est_center-one_gt_center).norm() <= 0.05){
                        est_centers.push_back(one_est_center);
                        if((one_gt_center-last_draw_pos).norm() > 0.02){
                            // draw frustum
                            glColor4f(1.f, 0.f, 0.f, 1.f);
                            pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, reloc_poses[rfid], 0.02f);
                            // pangolin::glDrawAxis(pose, 0.005f);

                            // draw distance
                            glColor4f(0.f, 0.f, 1.f, 1.f);
                            std::vector<Eigen::Matrix<float, 3, 1>> dist_centers;
                            dist_centers.push_back(one_gt_center);
                            dist_centers.push_back(one_est_center);
                            pangolin::glDrawVertices(dist_centers, GL_LINE_STRIP);
                            glColor4f(1.f, 1.f, 1.f, 1.f);
                            last_draw_pos = one_gt_center;
                        }

                        // if(rfid - last_draw_id >= 10){
                        //     // draw frustum
                        //     glColor4f(0.f, 0.f, 1.f, 1.f);
                        //     pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, reloc_poses[rfid], 0.02f);
                        //     // pangolin::glDrawAxis(pose, 0.005f);
                        //     // draw distance
                        //     glColor4f(1.f, 0.f, 0.f, 1.f);
                        //     std::vector<Eigen::Matrix<float, 3, 1>> dist_centers;
                        //     dist_centers.push_back(one_gt_center);
                        //     dist_centers.push_back(one_est_center);
                        //     pangolin::glDrawVertices(dist_centers, GL_LINE_STRIP);
                        //     glColor4f(1.f, 1.f, 1.f, 1.f);
                        //     last_draw_id = rfid;
                        // }
                    }
                }
                
                // draw the trajectory of the ground truth
                glColor4f(0.f, 1.f, 0.f, 1.f);
                pangolin::glDrawVertices(gt_centers, GL_LINE_STRIP);
                glColor4f(1.f, 1.f, 1.f, 1.f);
                gt_centers.clear();
                // // draw the trajectory of the estimates
                // if(est_centers.size() > 0)
                // {
                //     glColor4f(0.f, 0.f, 1.f, 1.f);
                //     pangolin::glDrawVertices(est_centers, GL_LINE_STRIP);
                //     glColor4f(1.f, 1.f, 1.f, 1.f);
                //     est_centers.clear();
                // }
                
            }
        }

    }
    if (*BoxDisplayCuboids)
    {
        // display cuboids
        // std::vector<std::pair<int, std::vector<float>>> label_dim_pairs = slam->get_object_cuboids(); 
        std::vector<std::pair<int, std::vector<float>>> label_dim_pairs = slam->get_objects(true);
        switch (*BarSwitchCuboid){
            case 1:
                for(size_t i=0; i<label_dim_pairs.size(); ++i)
                {
                    if(label_dim_pairs[i].first != 1)
                        continue;
                    DrawCuboids(label_dim_pairs, i, false);
                }
                break;
            case 2:
                for(size_t i=0; i<label_dim_pairs.size(); ++i)
                {
                    if(label_dim_pairs[i].first != 2)
                        continue;
                    DrawCuboids(label_dim_pairs, i, false);
                }
                break;
            case 3:
                for(size_t i=0; i<label_dim_pairs.size(); ++i)
                {
                    if(label_dim_pairs[i].first != 3)
                        continue;
                    DrawCuboids(label_dim_pairs, i, false);
                }
                break;
            case 4:
                for(size_t i=0; i<label_dim_pairs.size(); ++i)
                {
                    if(label_dim_pairs[i].first != 4)
                        continue;
                    DrawCuboids(label_dim_pairs, i, false);
                }
                break;
            case 5:
                for(size_t i=0; i<label_dim_pairs.size(); ++i)
                {
                    if(label_dim_pairs[i].first != 5)
                        continue;
                    DrawCuboids(label_dim_pairs, i, false);
                }
                break;
            case 6:
                for(size_t i=0; i<label_dim_pairs.size(); ++i)
                {
                    if(label_dim_pairs[i].first != 6)
                        continue;
                    DrawCuboids(label_dim_pairs, i, false);
                }
                break;
            case 7:
                for(size_t i=0; i<label_dim_pairs.size(); ++i)
                {
                    DrawCuboids(label_dim_pairs, i, false);
                }
                break;
            default:
                break;
        }
    }

    if (IsPaused())
    {
        if(*BoxDisplayAllCuboids)
        {
            std::vector<std::pair<int, std::vector<float>>> label_dim_pairs = slam->get_objects(false);
            switch (*BarSwitchCuboid){
                case 1:
                    for(size_t i=0; i<label_dim_pairs.size(); ++i)
                    {
                        if(label_dim_pairs[i].first != 1)
                            continue;
                        DrawCuboids(label_dim_pairs, i, false);
                    }
                    break;
                case 2:
                    for(size_t i=0; i<label_dim_pairs.size(); ++i)
                    {
                        if(label_dim_pairs[i].first != 2)
                            continue;
                        DrawCuboids(label_dim_pairs, i, false);
                    }
                    break;
                case 3:
                    for(size_t i=0; i<label_dim_pairs.size(); ++i)
                    {
                        if(label_dim_pairs[i].first != 3)
                            continue;
                        DrawCuboids(label_dim_pairs, i, false);
                    }
                    break;
                case 4:
                    for(size_t i=0; i<label_dim_pairs.size(); ++i)
                    {
                        if(label_dim_pairs[i].first != 4)
                            continue;
                        DrawCuboids(label_dim_pairs, i, false);
                    }
                    break;
                case 5:
                    for(size_t i=0; i<label_dim_pairs.size(); ++i)
                    {
                        if(label_dim_pairs[i].first != 5)
                            continue;
                        DrawCuboids(label_dim_pairs, i, false);
                    }
                    break;
                case 6:
                    for(size_t i=0; i<label_dim_pairs.size(); ++i)
                    {
                        if(label_dim_pairs[i].first != 6)
                            continue;
                        DrawCuboids(label_dim_pairs, i, false);
                    }
                    break;
                case 7:
                    for(size_t i=0; i<label_dim_pairs.size(); ++i)
                    {
                        DrawCuboids(label_dim_pairs, i, false);
                    }
                    break;
                default:
                    break;
            }
            
            // for(size_t i=0; i<label_dim_pairs.size(); ++i)
            // {
            //     DrawCuboids(label_dim_pairs, i, false);
            // }
        }

        if (*BoxDisplayKeyPoint)
        {
            slam->fetch_key_points(&keypoints[0], sizeKeyPoint, maxSizeKeyPoint);
            glColor4f(0.f, 1.f, 0.f, 1.f);
            glPointSize(3);
            pangolin::glDrawVertices(sizeKeyPoint, &keypoints[0], GL_POINTS, 3);
            glPointSize(1);
            glColor4f(1.f, 1.f, 1.f, 1.f);
        }
        if (*BoxDisplayPtCloud)
        {
            int num_objs = slam->get_num_objs();
            switch (*BarSwitchCuboid){
                case 1:
                    DrawPtClouds(num_objs, 1, false);
                    break;
                case 2:
                    DrawPtClouds(num_objs, 2, false);
                    break;
                case 3:
                    DrawPtClouds(num_objs, 3, false);
                    break;
                case 4:
                    DrawPtClouds(num_objs, 4, false);
                    break;
                case 5:
                    DrawPtClouds(num_objs, 5, false);
                    break;
                case 6:
                    DrawPtClouds(num_objs, 6, false);
                    break;
                case 7:
                    for(size_t i=0; i<num_objs; ++i){
                        float objPtsCloud[1000000] = {0};
                        size_t sizeObjPts=0;
                        int label = slam->get_object_pts(&objPtsCloud[0], sizeObjPts, i);
                        glColor4f(palette[label][0],
                                  palette[label][1], 
                                  palette[label][2],
                                  1.f);
                        glPointSize(2);
                        pangolin::glDrawVertices(sizeObjPts, &objPtsCloud[0], GL_POINTS, 3);
                        glPointSize(1);
                        glColor4f(1.f, 1.f, 1.f, 1.f);
                    }
                    break;
                default:
                    break;
            }
        }
        switch (*BarSwitchCuboidsReloc)
        {
            case 1:
            {    
                // in ground truth pose
                std::vector<std::pair<int, std::vector<float>>> kf_label_dim_pairs = slam->get_reloc_cuboids(1);
                switch (*BarSwitchCuboid){
                    case 1:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 1)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 1);
                        }
                        break;
                    case 2:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 2)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 1);
                        }
                        break;
                    case 3:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 3)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 1);
                        }
                        break;
                    case 4:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 4)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 1);
                        }
                        break;
                    case 5:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 5)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 1);
                        }
                        break;
                    case 6:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 6)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 1);
                        }
                        break;
                    case 7:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            DrawCuboids(kf_label_dim_pairs, i, true, 1);
                        }
                        break;
                    default:
                        break;
                }
                break;
            }
            case 2:
            {
                // in centroid reloc pose
                std::vector<std::pair<int, std::vector<float>>> kf_label_dim_pairs = slam->get_reloc_cuboids(2);
                switch (*BarSwitchCuboid){
                    case 1:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 1)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 2);
                        }
                        break;
                    case 2:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 2)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 2);
                        }
                        break;
                    case 3:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 3)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 2);
                        }
                        break;
                    case 4:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 4)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 2);
                        }
                        break;
                    case 5:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 5)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 2);
                        }
                        break;
                    case 6:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            if(kf_label_dim_pairs[i].first != 6)
                                continue;
                            DrawCuboids(kf_label_dim_pairs, i, true, 2);
                        }
                        break;
                    case 7:
                        for(size_t i=0; i<kf_label_dim_pairs.size(); ++i)
                        {
                            DrawCuboids(kf_label_dim_pairs, i, true, 2);
                        }
                        break;
                    default:
                        break;
                }
                break;
            }
            default:
                break;
        }
        if (*BoxDisplayPtCloudReloc)
        {
            // float *objPtsCloud;
            // size_t sizeObjPts=0, maxSizeObjPts=100000;
            // objPtsCloud = (float *)malloc(sizeof(float) * maxSizeObjPts);
            int num_objs = slam->get_reloc_num_objs();
            switch (*BarSwitchCuboid){
                case 1:
                    DrawPtClouds(num_objs, 1, true);
                    break;
                case 2:
                    DrawPtClouds(num_objs, 2, true);
                    break;
                case 3:
                    DrawPtClouds(num_objs, 3, true);
                    break;
                case 4:
                    DrawPtClouds(num_objs, 4, true);
                    break;
                case 5:
                    DrawPtClouds(num_objs, 5, true);
                    break;
                case 6:
                    DrawPtClouds(num_objs, 6, true);
                    break;
                case 7:
                    for(size_t i=0; i<num_objs; ++i){
                        float objPtsCloud[100000] = {0};
                        size_t sizeObjPts=0;
                        int label = slam->get_reloc_obj_pts(&objPtsCloud[0], sizeObjPts, i, bUseGT);
                        glColor4f(0.5f, 0.5f, 0.5f, 1.f);
                        glPointSize(2);
                        pangolin::glDrawVertices(sizeObjPts, &objPtsCloud[0], GL_POINTS, 3);
                        glPointSize(1);
                        glColor4f(1.f, 1.f, 1.f, 1.f);
                    }
                    break;
                default:
                    break;
            }
        }
        if(*BoxDisplayWorldOrigin){
            glColor4f(1.f, 0.f, 0.f, 1.f);
            glLineWidth(1);
            pangolin::glDrawLine(0,0,0, 0.1,0,0);
            glColor4f(0.f, 1.f, 0.f, 1.f);
            pangolin::glDrawLine(0,0,0, 0,0.1,0);
            glColor4f(0.f, 0.f, 1.f, 1.f);
            pangolin::glDrawLine(0,0,0, 0,0,0.1);
            glColor4f(1.f, 1.f, 1.f, 1.f);
        }
        // if (*BoxRecordData)
        // {
        //     slam->recordSequence("~/SLAM_work/Datasets/");
        // }
        
    }
    */
    
    pangolin::FinishFrame();

    auto t2 = std::chrono::system_clock::now();
    auto span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    if(span==0)
        span = 1;
    if(span<10000)
        usleep(10000-span);
}

void MainWindow::DrawMesh()
{
    auto vpMaps = slam->get_dense_maps();

    // ToDo: some future process to select the map to be visualised

    for (auto pMap : vpMaps)
    {
        if(!pMap->mbHasMesh)
            pMap->GenerateMesh();

        if (!pMap->mplPoint)
            continue;
        
        if(!pMap->mbVertexBufferCreated)
        {
            glGenBuffers(1, &pMap->mGlVertexBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, pMap->mGlVertexBuffer);
            // glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pMap->N, pMap->mplPoint, GL_STATIC_DRAW);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pMap->N, pMap->mplPoint, GL_DYNAMIC_DRAW);
            
            glGenBuffers(1, &pMap->mGlNormalBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, pMap->mGlNormalBuffer);
            // glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pMap->N, pMap->mplNormal, GL_STATIC_DRAW);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * pMap->N, pMap->mplNormal, GL_DYNAMIC_DRAW);
            
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            pMap->mbVertexBufferCreated = true;
        }

        Eigen::Matrix4f Tmw = pMap->GetPose().matrix().cast<float>();

        ShadingProg.Bind();
        ShadingProg.SetUniform("Tmw", pangolin::OpenGlMatrix(Tmw));
        ShadingProg.SetUniform("mvpMat", CameraView->GetProjectionModelViewMatrix());
        ShadingProg.SetUniform("colourTaint", pMap->mColourTaint);

        glBindBuffer(GL_ARRAY_BUFFER, pMap->mGlVertexBuffer);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, pMap->mGlNormalBuffer);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
        glEnableVertexAttribArray(1);

        glDrawArrays(GL_TRIANGLES, 0, pMap->N);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        ShadingProg.Unbind();
    }
}
void MainWindow::DeleteMesh()
{
    auto vpMaps = slam->get_dense_maps();
    for (auto pMap : vpMaps)
    {
        if(pMap->mbHasMesh)
            pMap->DeleteMesh();

        if(pMap->mbVertexBufferCreated)
        {
            glDeleteBuffers(1, &pMap->mGlVertexBuffer);
            glDeleteBuffers(1, &pMap->mGlNormalBuffer);
            pMap->mbVertexBufferCreated = false;
        }
    }
}

void MainWindow::UpdateMeshWithNormal()
{
    // // Main
    // auto *vertex = GetMappedVertexBuffer();
    // auto *normal = GetMappedNormalBuffer();
    // VERTEX_COUNT = slam->fetch_mesh_with_normal(vertex, normal);
}
void MainWindow::UpdateMeshWithColour()
{
    // auto *vertex = GetMappedVertexBuffer();
    // auto *colour = GetMappedColourBuffer();
    // VERTEX_COUNT = slam->fetch_mesh_with_colour(vertex, colour);
}

void MainWindow::DrawMeshShaded()
{
    // if (VERTEX_COUNT == 0)
    //     return;

    // ShadingProg.Bind();
    // glBindVertexArray(VAOShade);

    // ShadingProg.SetUniform("mvp_matrix", CameraView->GetProjectionModelViewMatrix());

    // glDrawArrays(GL_TRIANGLES, 0, VERTEX_COUNT * 3);

    // glBindVertexArray(0);
    // ShadingProg.Unbind();
}
void MainWindow::DrawMeshColoured()
{
    // if (VERTEX_COUNT == 0)
    //     return;

    // ShadingColorProg.Bind();
    // glBindVertexArray(VAOColour);

    // ShadingColorProg.SetUniform("mvp_matrix", CameraView->GetProjectionModelViewMatrix());

    // glDrawArrays(GL_TRIANGLES, 0, VERTEX_COUNT * 3);

    // glBindVertexArray(0);
    // ShadingColorProg.Unbind();
}
void MainWindow::DrawMeshNormalMapped()
{
}

float *MainWindow::GetMappedVertexBuffer()
{
    // return (float *)**MappedVertex;
}
float *MainWindow::GetMappedNormalBuffer()
{
    // return (float *)**MappedNormal;
}
unsigned char *MainWindow::GetMappedColourBuffer()
{
    // return (unsigned char *)**MappedColour;
}

/* Semantic & Reloc disabled for now.
void MainWindow::DrawCuboids(std::vector<std::pair<int, std::vector<float>>> label_dim_pairs,
                             int i, bool bRel, int usePose)
{
    int label = label_dim_pairs[i].first;
    std::vector<float> cuboid = label_dim_pairs[i].second;
    std::vector<float> main_axes;
    float color_val;
    if(!bRel){
        // main_axes = slam->get_obj_centroid_axes(i);
        // glColor4f(float(33554431 * label % 255)/255. , 
        //           float(32767 * label % 255)/255., 
        //           float(2097151 * label % 255)/255., 
        //           1.f);
        glColor4f(palette[label][0], 
                  palette[label][1], 
                  palette[label][2], 
                  1.f);

        // glColor4f(1.f, 1.f, 0.f, 1.f);
    } else {
        // main_axes = slam->get_reloc_obj_centroid_axes(i, usePose);
        // color_val = 0.5;
        // // glColor4f(0.5f, 0.5f, 0.5f, 1.f);

        // 1-GT, 2-reloc pose
        switch (usePose)
        {
        case 1:
            // glColor4f(0.f, 1.f, 0.f, 1.f);
            glColor4f(palette[label][0], 
                  palette[label][1], 
                  palette[label][2], 
                  1.f);
            break;
        case 2:
            glColor4f(1.f, 0.f, 0.f, 1.f);
            break;
        case 3:
            glColor4f(0.f, 0.f, 1.f, 1.f);
            break;
        default:
            break;
        }
    }
    glLineWidth(2);
    // corners with NOCS
    pangolin::glDrawLine(cuboid[0], cuboid[1], cuboid[2], cuboid[3], cuboid[4], cuboid[5]);
    pangolin::glDrawLine(cuboid[0], cuboid[1], cuboid[2], cuboid[6], cuboid[7], cuboid[8]);
    pangolin::glDrawLine(cuboid[0], cuboid[1], cuboid[2], cuboid[12], cuboid[13], cuboid[14]);
    pangolin::glDrawLine(cuboid[15], cuboid[16], cuboid[17], cuboid[3], cuboid[4], cuboid[5]);
    pangolin::glDrawLine(cuboid[15], cuboid[16], cuboid[17], cuboid[21], cuboid[22], cuboid[23]);
    pangolin::glDrawLine(cuboid[15], cuboid[16], cuboid[17], cuboid[12], cuboid[13], cuboid[14]);
    pangolin::glDrawLine(cuboid[9], cuboid[10], cuboid[11], cuboid[3], cuboid[4], cuboid[5]);
    pangolin::glDrawLine(cuboid[9], cuboid[10], cuboid[11], cuboid[6], cuboid[7], cuboid[8]);
    pangolin::glDrawLine(cuboid[9], cuboid[10], cuboid[11], cuboid[21], cuboid[22], cuboid[23]);
    pangolin::glDrawLine(cuboid[18], cuboid[19], cuboid[20], cuboid[6], cuboid[7], cuboid[8]);
    pangolin::glDrawLine(cuboid[18], cuboid[19], cuboid[20], cuboid[21], cuboid[22], cuboid[23]);
    pangolin::glDrawLine(cuboid[18], cuboid[19], cuboid[20], cuboid[12], cuboid[13], cuboid[14]);
    glLineWidth(1);
    // // glColor4f(1.f, 1.f, 1.f, 1.f);
    // // draw centroid
    // float centroid[3] = {main_axes[0], main_axes[1], main_axes[2]};
    // glPointSize(5);
    // pangolin::glDrawVertices(1, &centroid[0], GL_POINTS, 3);
    // glPointSize(1);
    // glColor4f(1.f, 1.f, 1.f, 1.f);
    // // draw main axes
    // glLineWidth(2);
    // glColor4f(color_val, 0.f, 0.f, 1.f);
    // pangolin::glDrawLine(main_axes[0], main_axes[1], main_axes[2], 
    //                         main_axes[9], main_axes[10], main_axes[11]);
    // glColor4f(0.f, color_val, 0.f, 1.f);
    // pangolin::glDrawLine(main_axes[0], main_axes[1], main_axes[2], 
    //                         main_axes[6], main_axes[7], main_axes[8]);
    // glColor4f(0.f, 0.f, color_val, 1.f);
    // pangolin::glDrawLine(main_axes[0], main_axes[1], main_axes[2], 
    //                         main_axes[3], main_axes[4], main_axes[5]);
    // glLineWidth(1);
    glColor4f(1.f, 1.f, 1.f, 1.f);
}

void MainWindow::DrawPtClouds(int num_objs, int idx, bool bRel)
{
    for(size_t i=0; i<num_objs; ++i){
        int label;
        size_t sizeObjPts=0;
        if(!bRel){
            float objPtsCloud[1000000] = {0};
            label = slam->get_object_pts(&objPtsCloud[0], sizeObjPts, i);
            glColor4f(palette[label][0], 
                  palette[label][1], 
                  palette[label][2], 
                  1.f);
            if(label!=idx)
                continue;
            glPointSize(2);
            pangolin::glDrawVertices(sizeObjPts, &objPtsCloud[0], GL_POINTS, 3);
        } else {
            float objPtsCloud[100000] = {0};
            label = slam->get_reloc_obj_pts(&objPtsCloud[0], sizeObjPts, i, bUseGT);
            glColor4f(0.5f, 0.5f, 0.5f, 1.f);
            if(label!=idx)
                continue;
            glPointSize(2);
            pangolin::glDrawVertices(sizeObjPts, &objPtsCloud[0], GL_POINTS, 3);
        }
        glPointSize(1);
        glColor4f(1.f, 1.f, 1.f, 1.f);
    }
}
*/

void MainWindow::SetCurrentCamera(Eigen::Matrix4f T)
{
    CameraPose = T;
    // pangolin::OpenGlMatrix ViewMat(CameraPose.inverse().eval());
    // CameraView->SetModelViewMatrix(ViewMat);
}

void MainWindow::SetSystem(fusion::System *sys)
{
    slam = sys;
}

