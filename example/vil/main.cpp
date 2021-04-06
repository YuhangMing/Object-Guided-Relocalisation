#include "system.h"
#include "visualization/main_window.h"
#include "utils/settings.h"
#include <ctime>

bool load_next_image_vil_sequence(cv::Mat &depth, cv::Mat &color, std::string img_path);
int image_counter;

int main(int argc, char **argv)
{
  if(argc < 2){
    std::cout << "executable sequence_number" << std::endl;
    return 0;
  }

  /* Parsing Process Options */
  int sequence_id = std::atoi(argv[1]);
  std::string img_path = GlobalCfg.data_path + "sequence0" + argv[1] + "/Construction";
//   std::string img_path = GlobalCfg.data_path + "sequence0" + argv[1] + "/Relocalisation";
  std::cout << "-- Loading data from " << img_path << std::endl;
  GlobalCfg.map_file += argv[1];
  if(GlobalCfg.bRecord){
    GlobalCfg.bEnableViewer = true;
    GlobalCfg.bSubmapping = false;
    GlobalCfg.bSemantic = false;
    GlobalCfg.bLoadDiskMap = false;
    GlobalCfg.bPureReloc = false;
    std::cout << "-- When recording, disable everything except visualisation." << std::endl;
  }
  if(!GlobalCfg.bLoadDiskMap && GlobalCfg.bPureReloc){
    // construct the map and then relocalisation
    std::cout << "Still need to figure out how to construct and reloc" << std::endl;
    return 1;
  }
  if(GlobalCfg.bOutputPose){
    GlobalCfg.output_pose_file = GlobalCfg.output_pose_file + "seq0" + argv[1] + ".txt";
    std::cout << "-- Output poses to " << GlobalCfg.output_pose_file << std::endl;
  }

  /* Begin Processing */
  image_counter = 0;
  cv::Mat image, depth;
  SetCalibration();
  fusion::System slam(0);
  std::cout << "\n------------------------------------------------" << std::endl;
  std::cout <<   "-------      Initialised SLAM system     -------" << std::endl;
  
  if(GlobalCfg.bEnableViewer)
  {
    MainWindow window("Object-Guided-Reloc", 1920, 920);
    window.SetSystem(&slam);
    std::cout <<   "------- Initialised visualisation window -------" << std::endl;
    std::cout <<   "-------      Entering the main loop      -------" << std::endl;
    std::cout <<   "------------------------------------------------" << std::endl;
    while (!pangolin::ShouldQuit())
    {
      if (!window.IsPaused())
      {
        if(load_next_image_vil_sequence(depth, image, img_path)) {
            // std::clock_t start = std::clock();
            if(GlobalCfg.bPureReloc)
            slam.relocalize_image(depth, image);
            else
            slam.process_images(depth, image);
            // std::cout << "## Processing an image takes " << ( std::clock() - start ) / (double) CLOCKS_PER_SEC << " seconds" << std::endl;
            
            if(!GlobalCfg.bRecord && GlobalCfg.bSemantic)
              window.SetRGBSource(slam.get_detected_image());
            else
              window.SetRGBSource(image);
            window.SetDepthSource(depth);   // raw depth
            // window.SetDetectedSource(slam.get_detected_image());
            // window.SetRenderScene(slam.get_rendered_scene());

            window.SetCurrentCamera(slam.get_camera_pose());
            window.mbFlagUpdateMesh = true;
        } else {
            window.SetPause();
        }
      }
      
      window.Render();
    }
  }
  else
  {
    std::cout <<   "-------      Entering the main loop      -------" << std::endl;
    std::cout <<   "------------------------------------------------" << std::endl;
    while(load_next_image_vil_sequence(depth, image, img_path)){
      if(GlobalCfg.bPureReloc)
          slam.relocalize_image(depth, image);
        else
          slam.process_images(depth, image);
    }
    // slam.writeMapToDisk("map-"+folder+"0"+std::to_string(sequence_id)+".data");
  }

  if(GlobalCfg.bOutputPose){
    slam.save_full_trajectory();
    std::cout << "Full trajectory saved to " << GlobalCfg.output_pose_file << std::endl;
  }
}

bool load_next_image_vil_sequence(cv::Mat &depth, cv::Mat &color, std::string img_path)
{
  // depth
  std::string name_depth = img_path + "/depth/" + std::to_string(image_counter) + ".png";
  depth = cv::imread(name_depth, cv::IMREAD_UNCHANGED);

  // color
  std::string name_color = img_path + "/color/" + std::to_string(image_counter) + ".png";
  color = cv::imread(name_color, cv::IMREAD_UNCHANGED);
  
  if(depth.empty() || color.empty()){
  	std::cout << "!!! Failed to load image " << image_counter << ".png. PAUSING..." << std::endl;
  	return false;
  }

  cv::cvtColor(color, color, CV_BGR2RGB);

  // cv::imshow("RGB", color);
  // cv::imshow("Depth", depth);
  // cv::waitKey(0);

  image_counter++;
	return true;
}
