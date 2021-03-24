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

    std::string img_path = GlobalCfg.data_folder + "sequence0" + argv[1] + "/Construction";
    int sequence_id = std::atoi(argv[1]);

    std::cout << "Running sequence0" << sequence_id << " with " 
            << GlobalCfg.num_img[sequence_id] << " frames loaded from "
            << img_path
            << std::endl;

    SetCalibration();
    
    fusion::System slam(GlobalCfg.bSemantic);

    image_counter = 0;
    cv::Mat image, depth;
    if(GlobalCfg.bEnableViewer){
        MainWindow window("Object-Guided-Reloc", 1920, 920);
        window.SetSystem(&slam);

        while (!pangolin::ShouldQuit())
        {
            if (!window.IsPaused() && load_next_image_vil_sequence(depth, image, img_path))
            {
                window.SetRGBSource(image);
                window.SetDepthSource(depth);   // raw depth
                // std::clock_t start = std::clock();
                slam.process_images(depth, image,
                        GlobalCfg.bSubmapping, GlobalCfg.bSemantic, GlobalCfg.bRecord);
                // std::cout << "## Processing an image takes "
                    //     << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
                    //     << " seconds" << std::endl;
                /* Semantic & Reloc Disabled for now
                window.SetDetectedSource(slam.get_detected_image());
                window.SetRenderScene(slam.get_rendered_scene());
                */
               
                window.SetCurrentCamera(slam.get_camera_pose());
                window.mbFlagUpdateMesh = true;

                // if(image_counter > num_img[sequence_id-1] || slam.b_reloc_attp)
                if(image_counter > GlobalCfg.num_img[sequence_id]){
                    std::cout << "ALL IMAGES LOADED !!!!" << std::endl;
                    window.SetPause();
                }
            }

            // if (window.IsPaused() && window.mbFlagUpdateMesh)
            if (window.mbFlagUpdateMesh)
            {
                auto *vertex = window.GetMappedVertexBuffer();
                auto *colour = window.GetMappedColourBuffer();
                window.VERTEX_COUNT = slam.fetch_mesh_with_colour(vertex, colour);

                window.mbFlagUpdateMesh = false;
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
	// if(image_counter > num_img[id]){
    //     std::cout << "!!! REACHED THE END OF THE SEQUENCE. " << std::endl;
    // 	return false;
    // }
    // if(image_counter == num_img[id])
    // 	std::cout << "LAST IMAGE LOADED !!!!" << std::endl;

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
