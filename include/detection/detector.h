#ifndef DETECTOR_H
#define DETECTOR_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <string>



namespace fusion
{

#define Detector_MaskRCNN 0
#define Detector_NOCS 1

namespace semantic
{

class Detector
{
public:
    Detector();
    ~Detector();

    virtual void initializeDetector(char* config_path, long val){
        std::cout << "Do nothing in the base class." << std::endl;
    }
	virtual void performDetection(cv::Mat image, cv::Mat depth=cv::Mat::zeros(2,2,CV_8U)){
        std::cout << "Do nothing in the base class." << std::endl;
    }
    virtual void releaseMemory(){
        std::cout << "Do nothing in the base class." << std::endl;
    }

    // c++ data structure
    // shared in MaskRCNN and NOCS
    int numDetection;
    int *pMasks;
    int *pLabels;
    float *pScores;
    // // only in MaskRCNN
    // float *pBoxes;
    // only in NOCS
    float *pCoord;

    std::vector<std::string> CATEGORIES;
    int detector_name;

protected:
    // PyObject
    // PyObject *pPyName, *pPyArgs, *pPyFunc, *pPyValue, *pPyImage;
	PyObject *pPyModule, *pPyModel;
    // PyArrayObject
    // PyArrayObject *pArrLabels, *pArrScores, *pArrMasks, *pArrCoords;

};

class MaskRCNN : public Detector {
public:
    MaskRCNN(char* module_name);
	~MaskRCNN();

	void initializeDetector(char* config_path, long val);
	void performDetection(cv::Mat image, cv::Mat depth=cv::Mat::zeros(2,2,CV_8U));
	void detectionWithLoad(char* img_path);

	// int getNumDetected();
	// // int* getDetectedImg();
    // // long int* getDetectedLabels();
	// // float* getDetectedScores();
	// // int* getDetectedMasks();
    // // OR
    // void getDetectedImg(int *pDetectedImg);
    // void getDetectedLabels(long int *pDetectedLabels);
	// void getDetectedScores(float *pDetectedScores);
	// void getDetectedMasks(int *pDetectedMasks);

    // c++ data structure
    int *pDetected;

private:
    // PyArrayObject
    PyArrayObject *pArrBoxes;
    // PyArrayObject *pArrDetected;
};

class NOCS : public Detector {
public:
    NOCS(char* module_name);
	~NOCS();

	void initializeDetector(char* config_path, long val);
	void performDetection(cv::Mat image, cv::Mat depth=cv::Mat::zeros(2,2,CV_8U));
	// void detectionWithLoad(char* img_path);
    void releaseMemory();

// private:
//     // PyArrayObject
//     PyArrayObject *pArrCoords;
};


} // namespace semantic

} // namespace fusion

#endif