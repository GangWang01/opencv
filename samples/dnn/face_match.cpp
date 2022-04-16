// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

#include "opencv2/objdetect.hpp"


using namespace cv;
using namespace std;

static Mat visualize(Mat input, Mat faces, int thickness = 2)
{
    Mat output = input.clone();
    for (int i = 0; i < faces.rows; i++)
    {
        // Print results
        cout << "Face " << i
            << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
            << "box width: " << faces.at<float>(i, 2) << ", box height: " << faces.at<float>(i, 3) << ", "
            << "score: " << faces.at<float>(i, 14) << "\n";

        // Draw bounding box
        rectangle(output, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(0, 255, 0), thickness);
        // Draw landmarks
        circle(output, Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, Scalar(255, 0, 0), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, Scalar(0, 0, 255), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, Scalar(0, 255, 0), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, Scalar(255, 0, 255), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, Scalar(0, 255, 255), thickness);
    }
    return output;
}

static std::string getFileName(std::string filePath, bool withExtension = true, char seperator = '\\')
{
    // Get last dot position
    std::size_t dotPos = filePath.rfind('.');
    std::size_t sepPos = filePath.rfind(seperator);
    if (sepPos != std::string::npos)
    {
        return filePath.substr(sepPos + 1, filePath.size() - (withExtension || dotPos != std::string::npos ? 1 : dotPos));
    }
    return "";
}

int main(int argc, char ** argv)
{
    if (argc != 5)
    {
        std::cerr << "Usage " << argv[0] << ": "
                  << "<det_onnx_path> "
                  << "<reg_onnx_path> "
                  << "<image1>"
                  << "<image2>\n";
        return -1;
    }

    String det_onnx_path = argv[1];
    String reg_onnx_path = argv[2];
    String image1_path = argv[3];
    String image2_path = argv[4];
    std::cout<<image1_path<<" "<<image2_path<<std::endl;
    Mat image1 = imread(image1_path);
    Mat image2 = imread(image2_path);

    float score_thresh = 0.9f;
    float nms_thresh = 0.3f;
    double cosine_similar_thresh = 0.363;
    double l2norm_similar_thresh = 1.128;
    int top_k = 5000;

    // Initialize FaceDetector
    Ptr<FaceDetectorYN> faceDetector;

    faceDetector = FaceDetectorYN::create(det_onnx_path, "", image1.size(), score_thresh, nms_thresh, top_k);
    Mat faces_1;
    faceDetector->detect(image1, faces_1);
    if (faces_1.rows < 1)
    {
        std::cerr << "Cannot find a face in " << image1_path << "\n";
        return -1;
    }

    faceDetector = FaceDetectorYN::create(det_onnx_path, "", image2.size(), score_thresh, nms_thresh, top_k);
    Mat faces_2;
    faceDetector->detect(image2, faces_2);
    if (faces_2.rows < 1)
    {
        std::cerr << "Cannot find a face in " << image2_path << "\n";
        return -1;
    }

    // Initialize FaceRecognizerSF
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(reg_onnx_path, "");
    Mat matchedFaces;
    for (int i = 0; i < faces_2.rows; i++)
    {
        Mat aligned_face1, aligned_face2;
        faceRecognizer->alignCrop(image1, faces_1.row(0), aligned_face1);
        faceRecognizer->alignCrop(image2, faces_2.row(i), aligned_face2);

        Mat feature1, feature2;
        faceRecognizer->feature(aligned_face1, feature1);
        feature1 = feature1.clone();
        faceRecognizer->feature(aligned_face2, feature2);
        feature2 = feature2.clone();

        double cos_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_COSINE);
        double L2_score = faceRecognizer->match(feature1, feature2, FaceRecognizerSF::DisType::FR_NORM_L2);

        bool cosFlag = false;
        bool L2Flag = false;
        if (cos_score >= cosine_similar_thresh)
        {
            cosFlag = true;
            std::cout << "They have the same identity;";
        }
        else
        {
            std::cout << "They have different identities;";
        }
        std::cout << " Cosine Similarity: " << cos_score << ", threshold: " << cosine_similar_thresh << ". (higher value means higher similarity, max 1.0)\n";

        if (L2_score <= l2norm_similar_thresh)
        {
            L2Flag = true;
            std::cout << "They have the same identity;";
        }
        else
        {
            std::cout << "They have different identities.";
        }
        std::cout << " NormL2 Distance: " << L2_score << ", threshold: " << l2norm_similar_thresh << ". (lower value means higher similarity, min 0.0)\n";

        if (cosFlag && L2Flag)
        {
            matchedFaces.push_back(faces_2.row(i));
        }
    }
    if (matchedFaces.rows > 0)
    {
        // Draw results on the input image
        Mat result = visualize(image2, matchedFaces);
        string outputName = "dnn_mat_" + getFileName(image2_path, false);
        //imwrite(outputName, result);
        namedWindow(image2_path, WINDOW_AUTOSIZE);
        imshow(image2_path, result);
        waitKey(0);
    }

    return 0;
}
