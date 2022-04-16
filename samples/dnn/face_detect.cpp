#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>

using namespace cv;
using namespace std;

static Mat visualize(Mat input, Mat faces, bool showMatch = false, string markText = "Target Found", int thickness=2)
{
    Mat output = input.clone();
    for (int i = 0; i < faces.rows; i++)
    {
        // Print results
        cout << "Face " << i
             << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
             << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
             << "score: " << faces.at<float>(i, 14) << "\n";

        // Draw bounding box
        rectangle(output, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(0, 255, 0), thickness);
        if (showMatch)
        {
            putText(output, markText, Point(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)) + 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }
        // Draw landmarks
        circle(output, Point2i(int(faces.at<float>(i, 4)),  int(faces.at<float>(i, 5))),  2, Scalar(255,   0,   0), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 6)),  int(faces.at<float>(i, 7))),  2, Scalar(  0,   0, 255), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 8)),  int(faces.at<float>(i, 9))),  2, Scalar(  0, 255,   0), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, Scalar(255,   0, 255), thickness);
        circle(output, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, Scalar(  0, 255, 255), thickness);
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

static Mat FindU(Mat sample, Mat target, string det_onnx_path, string reg_onnx_path)
{
    float score_thresh = 0.9f;
    float nms_thresh = 0.3f;
    double cosine_similar_thresh = 0.363;
    double l2norm_similar_thresh = 1.128;
    int top_k = 5000;

    // Initialize FaceDetector
    Ptr<FaceDetectorYN> faceDetector;

    faceDetector = FaceDetectorYN::create(det_onnx_path, "", sample.size(), score_thresh, nms_thresh, top_k);
    Mat faces_1;
    faceDetector->detect(sample, faces_1);
    if (faces_1.rows < 1)
    {
        std::cerr << "Cannot find a face in the sample.\n";
        return target;
    }

    faceDetector = FaceDetectorYN::create(det_onnx_path, "", target.size(), score_thresh, nms_thresh, top_k);
    Mat faces_2;
    faceDetector->detect(target, faces_2);
    if (faces_2.rows < 1)
    {
        std::cerr << "Cannot find a face in the target.\n";
        return target;
    }

    // Initialize FaceRecognizerSF
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(reg_onnx_path, "");
    Mat matchedFaces;
    for (int i = 0; i < faces_2.rows; i++)
    {
        Mat aligned_face1, aligned_face2;
        faceRecognizer->alignCrop(sample, faces_1.row(0), aligned_face1);
        faceRecognizer->alignCrop(target, faces_2.row(i), aligned_face2);

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
        //string name = "FindU";
        Mat result = visualize(target, matchedFaces, true);
        return result;
        //string outputName = "dnn_mat_" + getFileName(name, false);
        //imwrite(outputName, result);
        //namedWindow(name, WINDOW_AUTOSIZE);
        //imshow(name, result);
        //waitKey(0);
    }

    return target;
}

int main(int argc, char ** argv)
{
    CommandLineParser parser(argc, argv,
        "{help  h           |            | Print this message.}"
        "{input i           |            | Path to the input image. Omit for detecting on default camera.}"
        "{model m           | yunet.onnx | Path to the model. Download yunet.onnx in https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx.}"
        "{score_threshold   | 0.9        | Filter out faces of score < score_threshold.}"
        "{nms_threshold     | 0.3        | Suppress bounding boxes of iou >= nms_threshold.}"
        "{top_k             | 5000       | Keep top_k bounding boxes before NMS.}"
        "{save  s           | false      | Set true to save results. This flag is invalid when using camera.}"
        "{vis   v           | true       | Set true to open a window for result visualization. This flag is invalid when using camera.}"
        "{fmodel f          | face_recognizer_fast.onnx | Path to the face recognizer model.}"
        "{fsample fs        |            | Path to the sample image with single face.}"
    );
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return -1;
    }

    String modelPath = parser.get<String>("model");

    float scoreThreshold = parser.get<float>("score_threshold");
    float nmsThreshold = parser.get<float>("nms_threshold");
    int topK = parser.get<int>("top_k");

    bool save = parser.get<bool>("save");
    bool vis = parser.get<bool>("vis");

    String faceRecModelPath = parser.get<String>("fmodel");
    String faceSamplePath = parser.get<String>("fsample");

    // Initialize FaceDetectorYN
    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(modelPath, "", Size(320, 320), scoreThreshold, nmsThreshold, topK);

    // If input is an image
    if (parser.has("input"))
    {
        String input = parser.get<String>("input");
        Mat image = imread(input);

        // Set input size before inference
        detector->setInputSize(image.size());

        // Inference
        Mat faces;
        detector->detect(image, faces);

        // Draw results on the input image
        Mat result = visualize(image, faces);

        // Save results if save is true
        if(save)
        {
            string outputName = "dnn_det_" + getFileName(input, false);
            cout << "Results saved to " << outputName << "\n";
            imwrite(outputName, result);
        }

        // Visualize results
        if (vis)
        {
            namedWindow(input, WINDOW_AUTOSIZE);
            imshow(input, result);
            waitKey(0);
        }
    }
    else
    {
        int deviceId = 0;
        VideoCapture cap;
        cap.open(deviceId, CAP_ANY);
        int frameWidth = int(cap.get(CAP_PROP_FRAME_WIDTH));
        int frameHeight = int(cap.get(CAP_PROP_FRAME_HEIGHT));
        detector->setInputSize(Size(frameWidth, frameHeight));

        Mat frame;
        TickMeter tm;
        String msg = "FPS: ";
        while(waitKey(1) < 0) // Press any key to exit
        {
            // Get frame
            if (!cap.read(frame))
            {
                cerr << "No frames grabbed!\n";
                break;
            }

            // Inference
            Mat faces;
            tm.start();
            detector->detect(frame, faces);
            tm.stop();

            Mat result;
            if (parser.has("fmodel") && parser.has("fsample"))
            {
                // Recognize face
                result = FindU(imread(faceSamplePath), frame, modelPath, faceRecModelPath);
            }
            else
            {
                // Draw results on the input image
                result = visualize(frame, faces);
            }

            putText(result, msg + to_string(tm.getFPS()), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            // Visualize results
            imshow("Live", result);

            tm.reset();
        }
    }
}
