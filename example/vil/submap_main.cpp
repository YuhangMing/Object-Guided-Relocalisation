#include "system.h"
#include "visualization/main_window.h"
#include <ctime>
#include "utils/safe_call.h"

bool load_next_image_vil_sequence(cv::Mat &depth, cv::Mat &color, std::string img_path, int id);
int image_counter;
//- BOR DATASET
int num_img = 0;

int main(int argc, char **argv)
{
    if(argc < 4){
		std::cout << "executable data_path folder_name sequence_number" << std::endl;
		return 0;
	}

    bool bSubmapping = false;
    bool bSemantic = true;
    bool bRecord = false;
    std::string data_path = argv[1];
    std::string folder = argv[2];
    int sequence_id = std::atoi(argv[3]);
    cv::Mat image, depth;
    // load the map instead of construct from the begining
    image_counter = 0;
    
    std::cout << "Initializing SLAM..." << std::endl;
    safe_call(cudaGetLastError());
	fusion::IntrinsicMatrix K(640, 480, 580, 580, 319.5, 239.5);
    safe_call(cudaGetLastError());
    fusion::System slam(K, 5);

    // std::cout << "Loading poses..." << std::endl;
    // load GT, pose_reloc, geom_reloc
    slam.load_pose_info(folder, sequence_id);
    slam.set_frame_id(image_counter);
    bool bInitial = true;
    std::string map_path = data_path + folder + "/maps/map-" + folder + "0" + std::to_string(sequence_id) + ".data";
    std::string img_path = data_path + folder + "/sequence0" + std::to_string(sequence_id) + "/Relocalisation";
    std::cout << map_path << "\n" << img_path << std::endl;

    std::cout << "Initializing window..." << std::endl;
    MainWindow window("Object-Guided-Reloc", 1920, 920, false);
    window.SetSystem(&slam);

    while (!pangolin::ShouldQuit())
    {
        if (!window.IsPaused() && load_next_image_vil_sequence(depth, image, folder, sequence_id))
        {
            window.SetRGBSource(image);
            window.SetDepthSource(depth);
            // slam.process_images(depth, image, K, bSubmapping, bSemantic, bRecord);
            slam.relocalize_image(depth, image, K);
            if(bInitial){
                std::cout << "Loading the map" << std::endl;
                slam.readMapFromDisk(map_path);
                bInitial = false;
            }
            
            window.SetDetectedSource(slam.get_detected_image());
            window.SetRenderScene(slam.get_rendered_scene());
            window.SetNOCSMap(slam.get_NOCS_map());
            window.SetMask(slam.get_segmented_mask());
            window.SetCurrentCamera(slam.get_camera_pose());
            window.mbFlagUpdateMesh = true;

            if(image_counter > num_img || slam.b_reloc_attp)
                window.SetPause();

        }

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

bool load_next_image_vil_sequence(cv::Mat &depth, cv::Mat &color, std::string img_path, int id)
{
	if(image_counter > num_img){
        // std::cout << "!!! REACHED THE END OF THE SEQUENCE. " << std::endl;
    	return false;
    }
    if(image_counter == num_img)
    	std::cout << "LAST IMAGE LOADED !!!!" << std::endl;

	// std::string dir = "/home/yohann/SLAMs/datasets/"+folder+"/sequence0" + std::to_string(id) + "/Relocalisation";

    // depth
    std::string name_depth = img_path + "/depth/" + std::to_string(image_counter) + ".png";
    depth = cv::imread(name_depth, cv::IMREAD_UNCHANGED);

    // color
    std::string name_color = img_path + "/color/" + std::to_string(image_counter) + ".png";
    color = cv::imread(name_color, cv::IMREAD_UNCHANGED);
    cv::cvtColor(color, color, CV_BGR2RGB);
    
    image_counter++;

    if(depth.empty() || color.empty()){
    	std::cout << "!!! ERROR !!! Loading failed at image " << image_counter << std::endl;
    	return false;
    } else {
    	return true;
    }
}