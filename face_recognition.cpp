// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#include <fstream>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "face_gui.hpp"

using namespace dlib;
using namespace std;
// The next bit of code defines a ResNet network.  
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------
int main(int argc, char** argv) try
{

  std::vector<string> pred_names;

    //--------------------------------------- Face Database ---------------------------
    // Notice the database directory: default is "./database/trainingSet/trainingSet.txt"
    std::vector<string> face_names;
    std::vector<matrix<float,128,1>> face_data;
    {
      cout << "load faces database..." << endl;
      ifstream dBfile("../database/trainingSet/trainingSet.txt");
      string name; 
      matrix<float,128,1> face;
      while (dBfile >> name )
	{
	  face_names.push_back(name);
	  for(int i = 0; i <128; i++)
	    dBfile >> face(i, 0);
	  face_data.push_back(face);
	}
      cout << face_names.size() << "faces data loaded." << endl;
    }
    //-------------------------------------------------------------------------------------
    // Face detector.
    frontal_face_detector detector = get_frontal_face_detector();
    // Face landmarking model to align faces.
    shape_predictor sp;
    deserialize("../model_file/shape_predictor_5_face_landmarks.dat") >> sp;
    // DNN responsible for face recognition.
    anet_type net;
    deserialize("../model_file/dlib_face_recognition_resnet_model_v1.dat") >> net;

    // get the image: (1920, 1080) (1280, 720) (1024, 768) (960, 720) (640, 480) (320, 240)
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    if (!cap.isOpened()) {
	cerr << "Unable to connect to camera" << endl;
	return 1;
      }
    
    gui faceGui; // setup the GUI
    // Grab and process frames until the main window is closed by the user.
    while(!faceGui.is_closed())
      {
	cv::Mat temp;
	if (!cap.read(temp))break; 	// Grab a frame.
	
	cv::Mat temp_f; // Flipped image.
	cv::flip(temp, temp_f, 1);
	cv_image<bgr_pixel> img_show(temp_f); // Image for dispaly in GUI.

	cv::Mat temp_resized;  // Resized image for faster processing.
	cv::resize(temp_f, temp_resized, cv::Size(), 0.7, 0.7);
	cv_image<bgr_pixel> img(temp_resized);
	
	std::vector<rectangle> dets;
	

	dets = detector(img); // Run detector to detect faces.

	std::vector<matrix<rgb_pixel>> faces; // Normalized faces.
	for (auto face : dets){
	  auto shape = sp(img, face);
	  matrix<rgb_pixel> face_chip;
	  extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
	  faces.push_back(move(face_chip));
	}
	
	if (faces.size() == 0)
	  cout << "No faces found in image!" << endl;
	
	// This call asks the DNN to convert each face image in faces into a 128D vector.
	std::vector<matrix<float,0,1>> face_descriptors = net(faces);
	
	// This will find the correct person name based on the face_descriptors.
	pred_names.clear();
	float min_distance = 1.0, distance;
	size_t index_min_dist;
	for (size_t i = 0; i < face_descriptors.size(); ++i)
	  {
	    for (size_t j = 0; j < face_data.size(); ++j)
	      {
		distance = length(face_descriptors[i]-face_data[j]);
		if (distance < min_distance)
		  {
		    min_distance = distance;
		    index_min_dist = j;
		  }
	      }
	    pred_names.push_back(face_names[index_min_dist]);
	  }
	// Update the GUI given the predicted data.
	faceGui.updateGui(img_show, dets, pred_names);
      } //while

}
catch (std::exception& e)
{
    cout << e.what() << endl;
}
