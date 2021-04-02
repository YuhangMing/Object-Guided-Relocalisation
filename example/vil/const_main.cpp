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

    // std::string img_path = GlobalCfg.data_folder + "sequence0" + argv[1] + "/Construction";
    std::string img_path = GlobalCfg.data_folder + "sequence0" + argv[1] + "/Relocalisation";
    GlobalCfg.map_file += argv[1];  // update the map given current sequence id
    int sequence_id = std::atoi(argv[1]);

    std::cout << "Running sequence0" << sequence_id << " with " 
            << GlobalCfg.num_img[sequence_id] << " frames loaded from "
            << img_path
            << std::endl;

    SetCalibration();
    fusion::System slam(GlobalCfg.bSemantic, GlobalCfg.bLoadDiskMap);

    image_counter = 0;
    cv::Mat image, depth;
    if(GlobalCfg.bEnableViewer){
        MainWindow window("Object-Guided-Reloc", 1920, 920);
        window.SetSystem(&slam);
        std::cout << "Initialised: Visualisation window." << std::endl;

        std::cout << "\n------- Entering the main loop -------" << std::endl;
        while (!pangolin::ShouldQuit())
        {
            if (!window.IsPaused() && load_next_image_vil_sequence(depth, image, img_path))
            {
                // std::clock_t start = std::clock();
                slam.process_images(depth, image,
                        GlobalCfg.bSubmapping, GlobalCfg.bSemantic, GlobalCfg.bRecord);
                // std::cout << "## Processing an image takes "
                    //     << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
                    //     << " seconds" << std::endl;
                
                if(!GlobalCfg.bRecord && GlobalCfg.bSemantic)
                    window.SetRGBSource(slam.get_detected_image());
                else
                    window.SetRGBSource(image);
                window.SetDepthSource(depth);   // raw depth
                // window.SetDetectedSource(slam.get_detected_image());
                // window.SetRenderScene(slam.get_rendered_scene());

                window.SetCurrentCamera(slam.get_camera_pose());
                window.mbFlagUpdateMesh = true;

                // if(image_counter > num_img[sequence_id-1] || slam.b_reloc_attp)
                // if(image_counter > 100)
                if(image_counter > GlobalCfg.num_img[sequence_id])
                {
                    std::cout << "ALL IMAGES LOADED !!!!" << std::endl;
                    window.SetPause();
                }
            }
            
            window.Render();
        }
    }
    else
    {
        while(load_next_image_vil_sequence(depth, image, img_path)){        
            slam.process_images(depth, image,
                    GlobalCfg.bSubmapping, GlobalCfg.bSemantic, GlobalCfg.bRecord);
        }
        // slam.writeMapToDisk("map-"+folder+"0"+std::to_string(sequence_id)+".data");
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
    	std::cout << "!!! ERROR !!! Loading failed at image " << image_counter << std::endl;
    	return false;
    }

    cv::cvtColor(color, color, CV_BGR2RGB);

    // cv::imshow("RGB", color);
    // cv::imshow("Depth", depth);
    // cv::waitKey(0);

    image_counter++;
	return true;
}
