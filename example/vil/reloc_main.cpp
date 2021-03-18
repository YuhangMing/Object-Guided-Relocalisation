#include "system.h"
#include "visualization/main_window.h"
#include <ctime>

bool load_next_image_vil_sequence(cv::Mat &depth, cv::Mat &color, std::string img_path, int id);
int image_counter;
//- BOR sequences
int num_img = 300;  // number of the test frames

int main(int argc, char **argv)
{
    if(argc < 5){
		std::cout << "executable data_path folder_name sequence_number display_or_not" << std::endl;
		return 0;
	}

    bool bSubmapping = false;
    bool bSemantic = true;
    bool bRecord = false;
    std::string data_path = argv[1];
    std::string folder = argv[2];
    int sequence_id = std::atoi(argv[3]);
    std::string display = argv[4];
    cv::Mat image, depth;
    // load the map instead of construct from the begining
    image_counter = 0;

    std::cout << "Performing relocalisation in sequence0" << sequence_id << " with\n" 
              << "    " << num_img << " frames going to be test for reloc."
              << std::endl;
    
	fusion::IntrinsicMatrix K(640, 480, 580, 580, 319.5, 239.5);
    fusion::System slam(K, 5);
    // load GT, pose_reloc, geom_reloc
    slam.load_pose_info(folder, sequence_id);
    slam.set_frame_id(image_counter);
    slam.output_file_name = folder + "0" + std::to_string(sequence_id);
    bool bInitial = true;
    std::string map_path = data_path + folder + "/maps/map-" + folder + "0" + std::to_string(sequence_id) + ".data";
    std::string img_path = data_path + folder + "/sequence0" + std::to_string(sequence_id) + "/Relocalisation";
    std::cout << map_path << "\n" << img_path << std::endl;

    // "/home/yohann/SLAMs/datasets/BOR/maps/map-" + folder + "0" + std::to_string(sequence_id) + ".data";

    if(display == "true"){
        MainWindow window("Object-Guided-Reloc", 1920, 920, false);
        window.SetSystem(&slam);

        while (!pangolin::ShouldQuit())
        {
            if (!window.IsPaused() && load_next_image_vil_sequence(depth, image, img_path, sequence_id))
            {
                window.SetRGBSource(image);
                window.SetDepthSource(depth);

                // std::cout << "processing new image" << std::endl;
                // slam.process_images(depth, image, K, bSubmapping, bSemantic, bRecord);
                slam.relocalize_image(depth, image, K);
                if(bInitial){
                    std::cout << "Loading the map" << std::endl;
                    slam.readMapFromDisk(map_path);
                    bInitial = false;
                }
                // std::cout << "finished current relocalising" << std::endl;
                
                window.SetDetectedSource(slam.get_detected_image());
                window.SetRenderScene(slam.get_rendered_scene());
                // window.SetNOCSMap(slam.get_NOCS_map());
                // window.SetMask(slam.get_segmented_mask());
                window.SetCurrentCamera(slam.get_camera_pose());
                window.mbFlagUpdateMesh = true;
                // std::cout << "updated all window displays" << std::endl;

                if(image_counter > num_img || slam.pause_window)
                    window.SetPause();

            }

            if (window.mbFlagUpdateMesh)
            {
                auto *vertex = window.GetMappedVertexBuffer();
                auto *colour = window.GetMappedColourBuffer();
                window.VERTEX_COUNT = slam.fetch_mesh_with_colour(vertex, colour);

                window.mbFlagUpdateMesh = false;
            }

            // std::cout << "render the window again" << std::endl;
            window.Render();
            // std::cout << "render done" << std::endl;
        }
    }
    else
    {
        
        while(load_next_image_vil_sequence(depth, image, img_path, sequence_id)){        
            // slam.process_images(depth, image, K, bSubmapping, bSemantic, bRecord);
            slam.relocalize_image(depth, image, K);
            if(bInitial){
                slam.readMapFromDisk(map_path);
                bInitial = false;
            }
        }
    }
}

bool load_next_image_vil_sequence(cv::Mat &depth, cv::Mat &color, std::string img_path, int id)
{
	if(image_counter > num_img){
        std::cout << "!!! REACHED THE END OF THE SEQUENCE. " << std::endl;
    	return false;
    }
    if(image_counter == num_img)
    	std::cout << "LAST IMAGE LOADED !!!!" << std::endl;

	// std::string dir = path + folder + "/sequence0" + std::to_string(id) + "/Relocalisation";
    // "/home/yohann/SLAMs/datasets/"+folder+"/sequence0" + std::to_string(id) + "/Relocalisation";

    // depth
    std::string name_depth = img_path + "/depth/" + std::to_string(image_counter) + ".png";
    depth = cv::imread(name_depth, cv::IMREAD_UNCHANGED);
    // normalize for the purpose of visualization
    // cv::Mat scaledDepth;
    // normalize(depth, scaledDepth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // // scaledDepth.convertTo(scaledDepth, CV_8U, 1);
    // cv::imshow("Depth", scaledDepth);
    // int k = cv::waitKey(0);

    // color
    std::string name_color = img_path + "/color/" + std::to_string(image_counter) + ".png";
    color = cv::imread(name_color, cv::IMREAD_UNCHANGED);
    cv::cvtColor(color, color, CV_BGR2RGB);
    
    image_counter++;
    // image_counter += 111;

    if(depth.empty() || color.empty()){
    	std::cout << "!!! ERROR !!! Loading failed at image " << image_counter << std::endl;
    	return false;
    } else {
    	return true;
    }
}