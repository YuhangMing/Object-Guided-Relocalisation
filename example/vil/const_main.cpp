#include "system.h"
#include "visualization/main_window.h"
#include <ctime>

bool load_next_image_vil_sequence(cv::Mat &depth, cv::Mat &color, std::string img_path, int id);
int image_counter;
//- BOR sequences
int num_img[10] = {3522, 1237, 887, 1221, 809, 300, 919, 1470, 501, 870};      // num of frames for construction

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
    image_counter = 0;
    cv::Mat image, depth;

    std::cout << "Performing relocalisation in sequence0" << sequence_id << " with\n" 
            << "    " << num_img[sequence_id] << " frames used for recons."
            << std::endl;

	fusion::IntrinsicMatrix K(640, 480, 580, 580, 319.5, 239.5);
    fusion::System slam(K, 5, bSemantic);
    // load GT, pose_reloc, geom_reloc
    slam.load_pose_info(folder, sequence_id);

    std::string img_path = data_path + folder + "/sequence0" + std::to_string(sequence_id) + "/Relocalisation";
    std::cout << img_path << std::endl;

    if(display == "true"){
        MainWindow window("Object-Guided-Reloc", 1920, 920);
        window.SetSystem(&slam);

        while (!pangolin::ShouldQuit())
        {
            if (!window.IsPaused() && load_next_image_vil_sequence(depth, image, img_path, sequence_id))
            {
                window.SetRGBSource(image);
                window.SetDepthSource(depth);   // raw depth
                // std::clock_t start = std::clock();
                slam.process_images(depth, image, K, bSubmapping, bSemantic, bRecord);
                /*
                For key frames, it takes 0.8s~1s to process
                    object detection: ~0.4s
                    SURF points detection: ~0.4s
                For other frames, it usually takes 0.02s, but sometimes could take around 0.1s
                */
                // std::cout << "## Processing an image takes "
                    //     << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
                    //     << " seconds" << std::endl;
                
                window.SetDetectedSource(slam.get_detected_image());
                // window.SetDepthSource(slam.get_shaded_depth());      // rendered depth
                window.SetRenderScene(slam.get_rendered_scene());
                // window.SetRenderScene(slam.get_rendered_scene_textured());
                // window.SetNOCSMap(slam.get_NOCS_map());
                // window.SetMask(slam.get_segmented_mask());
                window.SetCurrentCamera(slam.get_camera_pose());
                window.mbFlagUpdateMesh = true;

                // if(image_counter > num_img[sequence_id-1] || slam.b_reloc_attp)
                if(image_counter > num_img[sequence_id])
                    window.SetPause();

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
        while(load_next_image_vil_sequence(depth, image, folder, sequence_id)){        
            slam.process_images(depth, image, K, bSubmapping, bSemantic, bRecord);
        }
        // slam.writeMapToDisk("map-"+folder+"0"+std::to_string(sequence_id)+".data");
    }
    
}

bool load_next_image_vil_sequence(cv::Mat &depth, cv::Mat &color, std::string img_path, int id)
{
	if(image_counter > num_img[id]){
        std::cout << "!!! REACHED THE END OF THE SEQUENCE. " << std::endl;
    	return false;
    }
    if(image_counter == num_img[id])
    	std::cout << "LAST IMAGE LOADED !!!!" << std::endl;

    // depth
    std::string name_depth = img_path + "/depth/" + std::to_string(image_counter) + ".png";
    depth = cv::imread(name_depth, cv::IMREAD_UNCHANGED);

    // color
    std::string name_color = img_path + "/color/" + std::to_string(image_counter) + ".png";
    color = cv::imread(name_color, cv::IMREAD_UNCHANGED);
    cv::cvtColor(color, color, CV_BGR2RGB);

    // cv::imshow("RGB", color);
    // cv::imshow("Depth", depth);
    // cv::waitKey(0);

    image_counter++;
    if(depth.empty() || color.empty()){
    	std::cout << "!!! ERROR !!! Loading failed at image " << image_counter << std::endl;
    	return false;
    } else {
    	return true;
    }
}