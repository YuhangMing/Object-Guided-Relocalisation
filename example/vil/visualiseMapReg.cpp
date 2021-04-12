#include "system.h"
#include "visualization/main_window.h"
#include "utils/settings.h"
#include <ctime>

int main(int argc, char **argv)
{
  if(argc < 2){
    std::cout << "executable tgt_seq src_seq src_img" << std::endl;
    return 0;
  }

  /* Parsing Process Options */
  std::string img_path_tgt = GlobalCfg.data_path + "sequence0" + argv[1] + "/color/150.png";
  std::string img_path_src = GlobalCfg.data_path + "sequence0" + argv[2] + "/color/" + argv[3] + ".png";;
  std::cout << "-- Images loaded from " << std::endl
            << "   " << img_path_tgt << std::endl
            << "   " << img_path_src << std::endl;
  GlobalCfg.map_file += argv[1];
  // enable visualisation window
  if(!GlobalCfg.bEnableViewer)
    GlobalCfg.bEnableViewer = true;
  // disable semantic
  if(GlobalCfg.bSemantic)
    GlobalCfg.bSemantic = false;
  // load map from disk
  if(!GlobalCfg.bLoadDiskMap)
    GlobalCfg.bLoadDiskMap = true;

  /* Begin Processing */
  cv::Mat img_tgt = cv::imread(img_path_tgt, cv::IMREAD_UNCHANGED);
  cv::Mat img_src = cv::imread(img_path_src, cv::IMREAD_UNCHANGED);
  if(img_tgt.empty() || img_src.empty()){
  	std::cout << "!!! Failed to load image. QUIT..." << std::endl;
  	return 1;
  }
  cv::cvtColor(img_tgt, img_tgt, CV_BGR2RGB);
  cv::cvtColor(img_src, img_src, CV_BGR2RGB);

  SetCalibration();
  fusion::System slam(0);
  slam.readOneMap("map3");
  std::cout << "\n------------------------------------------------" << std::endl;
  std::cout <<   "-------      Initialised SLAM system     -------" << std::endl;
  
  if(GlobalCfg.bEnableViewer)
  {
    MainWindow window("Map Registration", 1920, 920);
    window.SetSystem(&slam);
    std::cout <<   "------- Initialised visualisation window -------" << std::endl;
    std::cout <<   "-------      Entering the main loop      -------" << std::endl;
    std::cout <<   "------------------------------------------------" << std::endl;
    while (!pangolin::ShouldQuit())
    {
      window.SetRGBSource(img_tgt);
      window.SetDetectedSource(img_src);

      if (!window.IsPaused())
      {
        window.SetPause();
      }
      
      window.Render();
    }
  }
}

