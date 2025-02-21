/*
 * Example made with love by Natxopedreira 2021
 * https://github.com/natxopedreira
 * Updated by members of the ZKM | Hertz-Lab 2021
 */

#include "ofApp.h"

void makeDir(std::string dirName){
   

    // Create an ofDirectory object
    ofDirectory dir;

    // Set the path to the new directory
    dir = ofDirectory(ofToDataPath(dirName));

    // Check if the directory already exists
    if(!dir.exists()){
        // Create the directory(s) if it doesn't exist
        bool success = dir.create(true);
        if(success){
            ofLogNotice() << "Directory " << dirName << " created successfully!";
        } else {
            ofLogError() << "Failed to create directory " << dirName;
        }
    } else {
        ofLogNotice() << "Directory " << dirName << " already exists!";
    }
}
void drawPointsOpenFrameworksStyle(const std::vector<cv::Point>& points) {
    // Begin shape
    ofBeginShape();

    // Iterate over each point in the vector and draw it as a vertex
    for (const auto& point : points) {
        ofVertex(point.x, point.y);
    }

    // End shape
    ofEndShape();
}

ofMatrix4x4 cvMatToOFMatrix(const cv::Mat& cvMat) {
    // Create an empty ofMatrix4x4
    ofMatrix4x4 ofMat;
    
    // Check if the input matrix is empty
    if (cvMat.empty()) {
        // Return an empty ofMatrix4x4 if cv::Mat is empty
        return ofMat;
    }
    
    // Ensure the cv::Mat is a 2x3 matrix of type CV_32F
    if (cvMat.rows == 2 && cvMat.cols == 3) {
        // Initialize the 4x4 ofMatrix4x4 with default values
        ofMat.makeIdentityMatrix();

        // Copy values from the 2x3 matrix to the 4x4 matrix
        ofMat(0, 0) = cvMat.at<float>(0, 0);
        ofMat(0, 1) = cvMat.at<float>(0, 1);
        ofMat(0, 3) = cvMat.at<float>(0, 2);

        ofMat(1, 0) = cvMat.at<float>(1, 0);
        ofMat(1, 1) = cvMat.at<float>(1, 1);
        ofMat(1, 3) = cvMat.at<float>(1, 2);

        // The third row is (0, 0, 1) for affine transformations
        ofMat(2, 2) = 1.0f;
        ofMat(3, 3) = 1.0f;
    } else {
        // Throw an exception or handle error if the matrix is not 2x3 or not of type CV_32F
        throw std::invalid_argument("Input cv::Mat must be a 2x3 matrix of type CV_32F.");
    }

    return ofMat;
}
//
//std::vector<cv::Point> resampleContour(const std::vector<cv::Point>& contour, int newSize) {
//    std::vector<cv::Point> resampled;
//    int originalSize = contour.size();
//
//    if (newSize <= 0 || originalSize == 0) {
//        // Return an empty vector if the new size is invalid or the original contour is empty
//        return resampled;
//    }
//
//    if (newSize == originalSize) {
//        // If the new size is the same as the original, return the original contour
//        return contour;
//    }
//
//    // Calculate the cumulative arc length along the contour
//    std::vector<float> arcLength(originalSize);
//    arcLength[0] = 0;
//    for (int i = 1; i < originalSize; ++i) {
//        arcLength[i] = arcLength[i - 1] + cv::norm(contour[i] - contour[i - 1]);
//    }
//    float totalLength = arcLength.back();
//
//    // Generate new equally spaced arc lengths
//    std::vector<float> newArcLengths(newSize);
//    for (int i = 0; i < newSize; ++i) {
//        newArcLengths[i] = (i * totalLength) / (newSize - 1);
//    }
//
//    // Resample the contour
//    resampled.push_back(contour[0]); // First point remains the same
//    int nextPointIndex = 1;
//    for (int i = 1; i < newSize - 1; ++i) {
//        while (newArcLengths[i] > arcLength[nextPointIndex] && nextPointIndex < originalSize - 1) {
//            nextPointIndex++;
//        }
//
//        float t = (newArcLengths[i] - arcLength[nextPointIndex - 1]) / (arcLength[nextPointIndex] - arcLength[nextPointIndex - 1]);
//        cv::Point newPoint = contour[nextPointIndex - 1] * (1 - t) + contour[nextPointIndex] * t;
//        resampled.push_back(newPoint);
//    }
//    resampled.push_back(contour.back()); // Last point remains the same
//
//    return resampled;
//}



std::vector<cv::Point> resampleContour(const std::vector<cv::Point>& contour, int newSize) {
    std::vector<cv::Point> resampled;
    int originalSize = contour.size();

    if (newSize >= originalSize) {
        // No need to resample if the new size is greater than or equal to the original
        return contour;
    }

    // Calculate the cumulative arc length along the contour
    std::vector<float> arcLength(originalSize);
    arcLength[0] = 0;
    for (int i = 1; i < originalSize; ++i) {
        arcLength[i] = arcLength[i - 1] + cv::norm(contour[i] - contour[i - 1]);
    }
    float totalLength = arcLength.back();

    // Generate new equally spaced arc lengths
    std::vector<float> newArcLengths(newSize);
    for (int i = 0; i < newSize; ++i) {
        newArcLengths[i] = (i * totalLength) / (newSize - 1);
    }

    // Resample the contour
    resampled.push_back(contour[0]); // First point remains the same
    int nextPointIndex = 1;
    for (int i = 1; i < newSize - 1; ++i) {
        while (newArcLengths[i] > arcLength[nextPointIndex] && nextPointIndex < originalSize - 1) {
            nextPointIndex++;
        }

        float t = (newArcLengths[i] - arcLength[nextPointIndex - 1]) / (arcLength[nextPointIndex] - arcLength[nextPointIndex - 1]);
        cv::Point newPoint = contour[nextPointIndex - 1] * (1 - t) + contour[nextPointIndex] * t;
        resampled.push_back(newPoint);
    }
    resampled.push_back(contour.back()); // Last point remains the same

    return resampled;
}

std::pair<std::vector<cv::Point>, std::vector<cv::Point>> resampleContours(
    const std::vector<cv::Point>& contour1,
    const std::vector<cv::Point>& contour2) {
    
    if (contour1.size() == contour2.size()) {
        return {contour1, contour2};
    }

    const std::vector<cv::Point>* largerContour;
    const std::vector<cv::Point>* smallerContour;
    
    if (contour1.size() > contour2.size()) {
        largerContour = &contour1;
        smallerContour = &contour2;
    } else {
        largerContour = &contour2;
        smallerContour = &contour1;
    }

    // Resample the larger contour to match the size of the larger contour
    std::vector<cv::Point> resampledLargerContour = resampleContour(*largerContour, smallerContour->size());

    if (contour1.size() > contour2.size()) {
        return {resampledLargerContour, contour2};
    } else {
        return {contour1, resampledLargerContour};
    }
}

float calculateRotationFromContours(const std::vector<cv::Point>& srcContour, const std::vector<cv::Point>& dstContour) {
    // Calculate moments for both contours
    cv::Moments srcMoments = cv::moments(srcContour);
    cv::Moments dstMoments = cv::moments(dstContour);

    // Calculate the angle of orientation for each shape
    double srcAngle = 0.5 * atan2(2 * srcMoments.mu11, srcMoments.mu20 - srcMoments.mu02);
    double dstAngle = 0.5 * atan2(2 * dstMoments.mu11, dstMoments.mu20 - dstMoments.mu02);

    // Convert angles from radians to degrees
    srcAngle = srcAngle * 180.0 / CV_PI;
    dstAngle = dstAngle * 180.0 / CV_PI;

    // Calculate the rotation needed to align srcContour with dstContour
    float rotationNeeded = dstAngle - srcAngle;

    return rotationNeeded;
}

//--------------------------------------------------------------
void ofApp::setup() {
    
    ofSetFrameRate(30);
    ofSetVerticalSync(true);
    ofSetWindowTitle("olympicSans");



//    if(!model.load("model")) {
//        std::exit(EXIT_FAILURE);
//    }
//
//    std::vector<std::string> inputNames = {
//        "serving_default_downsample_ratio:0",
//        "serving_default_r1i:0",
//        "serving_default_r2i:0",
//        "serving_default_r3i:0",
//        "serving_default_r4i:0",
//        "serving_default_src:0"
//    };
//    std::vector<std::string> outputNames = {
//        "StatefulPartitionedCall:0",
//        "StatefulPartitionedCall:1",
//        "StatefulPartitionedCall:2",
//        "StatefulPartitionedCall:3",
//        "StatefulPartitionedCall:4",
//        "StatefulPartitionedCall:5"
//    };
//    model.setup(inputNames, outputNames);

    // parameters for the neural network
    float downsampleRatio = 0.25f;
    float videoWidth = 1920;
    float videoHeight = 1080;
    float batchSize = 1.0f;
    float numChannels = 3.0f;
    
    // model-specific inputs
//    inputs = {
//        cppflow::tensor({downsampleRatio}),
//        cppflow::tensor({0.0f}),                         // r1i
//        cppflow::tensor({0.0f}),                         // r2i
//        cppflow::tensor({0.0f}),                         // r3i
//        cppflow::tensor({0.0f}),                         // r4i
//        cppflow::tensor({1,1920,1080,3}) // batch size width height channels
//    };

    imgMask.allocate(videoWidth, videoHeight, OF_IMAGE_GRAYSCALE);
    grayImage.allocate(videoWidth, videoHeight);
    colorImage.allocate(videoWidth, videoHeight);// naturally , GL_RGB
    characterImage.allocate(videoWidth, videoHeight);
//    maskFbo.allocate(videoWidth, videoHeight);
//    videoFbo.allocate(videoWidth, videoHeight);
    
    fbo.allocate(videoWidth, videoHeight);
    pixels.allocate(1920,1080, OF_IMAGE_COLOR);
//    syphonPixels.allocate(1920,1080, OF_IMAGE_COLOR_ALPHA);
//    syphonImage.allocate(1920,1080, OF_IMAGE_COLOR);
    imgOut.allocate(videoWidth,videoHeight, OF_IMAGE_COLOR_ALPHA);

//    imgMask.allocate(videoWidth, videoHeight, OF_IMAGE_GRAYSCALE);
//    grayImage.allocate(videoWidth, videoHeight);
//    colorImage.allocate(videoWidth, videoHeight);
    characterImage.allocate(videoWidth, videoHeight);
    shapeMaskFbo.allocate(videoWidth, videoHeight, GL_RGB);
    videoFbo.allocate(videoWidth, videoHeight);\
    sourceFbo.allocate(camWidth, camHeight * 2);

    
    imgOut.getTexture().setAlphaMask(imgMask.getTexture());
    
//    testFont.load("pixel.ttf", 90, true, true, true);
    
    testFont.load("Paris2024-Variable.ttf", 90, true, true, true);

    letter = '_';
    letterPlace = 0;
    
    for (int v = 0 ; v < numFramesToGather ; v++){
        saveImgs[v].allocate(1920, 1080, OF_IMAGE_COLOR);
        curVideoFrames[v].allocate(1920,1080, OF_IMAGE_COLOR);
    }
    
//    // Fill the array with capital letters
    for(int i = 0; i < numChars; ++i) {
        bestFits[i].letter ='A' + i;
        
        bool vflip = true;
        bool filled = true;
        testChar = testFont.getCharacterAsPoints(bestFits[i].letter, vflip, filled);

        filled = false;
        testCharContour = testFont.getCharacterAsPoints(bestFits[i].letter, vflip, filled);

        //std::vector<std::vector<cv::Point>> cvBlobs2;
        std::vector<cv::Point> cvBlob2;
        for(int k = 0; k <(int)testCharContour.getOutline().size(); k++){
            if( k!= 0)ofNextContour(true) ;
//            std::vector<cv::Point> cvBlob2;
            for(int ii = 0; ii < (int)testCharContour.getOutline()[k].size(); ii++){
                
                    cvBlob2.emplace_back(cv::Point(static_cast<int>(testCharContour.getOutline()[k].getVertices()[ii].x), static_cast<int>(testCharContour.getOutline()[k].getVertices()[ii].y)));
               // ofLog()<<"emplace2";
                }
               // cvBlobs2.push_back(cvBlob2);
           // ofLog()<<"push2";
            }
        
        
        // this is for
        
        // A 0
        // a 32
        // Q 16
        // q 48
        // d 35
        // g 38
//        if (i == 16) {
//            bestFits[i].charContours = cvBlobs2[2];
        
        // TODO FIX THIS
//        if (i == 16) {
//            bestFits[i].charContours = cvBlob2[2];
//        }
//       else if (i == 0 || i == 32 || i == 16 || i == 48 || i == 35 || i == 38) {
//            bestFits[i].charContours = cvBlobs2[1];
//        }
//        else{
//            bestFits[i].charContours = cvBlobs2[0];
//        }
//
//        else{
            
            bestFits[i].charContours = cvBlob2; // one big
//         }
        
        for (int v = 0 ; v < numFramesToGather ; v++){
            bestFits[i].videoFrames[v].allocate(1920, 1080, OF_IMAGE_COLOR);
        }
    }
    

    ofSetFullscreen(false);
    //pix.allocate(video.getWidth(), video.getHeight());
    
    
        #ifdef USE_LIVE_VIDEO
    //        // setup video grabber
        mClient.setup();
        dir.setup();
        mClient.set(dir.getDescription(dir.size() - 1));
        #else
    ////        video.load("gymniastcs_trim.mp4");
    video.load("combo/swim00089.mp4");
    video.play();
    video.setVolume(0);
//    maskFbo.allocate(videoWidth, videoHeight/2);
//    videoFbo.allocate(videoWidth, videoHeight/2);

    // Allocate pixels and textures
    topPixels.allocate(camWidth, camHeight, OF_PIXELS_RGB);
    bottomPixels.allocate(camWidth, camHeight, OF_PIXELS_RGB);
    topTex.allocate(camWidth, camHeight, GL_RGB);
    bottomTex.allocate(camWidth, camHeight, GL_RGB);
    ////
//            video.setLoopState(OF_LOOP_NONE);
        #endif

    //using Syphon app Simple Server, found at http://syphon.v002.info/
  //  mClient.set("","OBS");

    // Add these new members at the start of setup
    ofDirectory comboDir("combo");
    comboDir.listDir();
    comboDir.allowExt("mp4");
    comboDir.sort();
    videoFiles = comboDir.getFiles();
    currentVideoIndex = 0;

    if (videoFiles.size() > 0) {
        video.load(videoFiles[currentVideoIndex].getAbsolutePath());
        video.play();
        video.setVolume(0);
    }
}

//--------------------------------------------------------------
void ofApp::update() {
    #ifdef USE_LIVE_VIDEO
    if(mClient.isSetup( ) == false){
        return ;
    }
    
    
    fbo.begin();
    mClient.draw(0,0,ofGetWidth(), ofGetHeight() );
    fbo.end();
    
    fbo.readToPixels(pixels);
    
    #else
    video.update();
 
    if(video.isFrameNew()) {
        // Calculate the half height for splitting the video
        // Calculate the full height and half height
               float fullHeight = 2160;  // Total video height (2x1080)
               float halfHeight = 1080;  // Height of each section
               
               // Draw top half (mask) to maskFbo
               shapeMaskFbo.begin();
               ofClear(0);
               video.draw(0, 0, camWidth, fullHeight);  // Draw full video, offset up to show top half
               shapeMaskFbo.end();
               
               // Draw bottom half (video) to videoFbo
               videoFbo.begin();
               ofClear(0);
               video.draw(0, -fullHeight/2., camWidth, fullHeight);  // Draw full video, showing bottom half
               videoFbo.end();

               //if(gatherFrames < numFramesToGather) {
                   videoFbo.readToPixels(curVideoFrames[gatherFrames % numFramesToGather]);
                   curVideoFrames[gatherFrames % numFramesToGather].setImageType(OF_IMAGE_COLOR);
               //}

//               // Process mask for contour detection
//               maskFbo.readToPixels(colorImage);
//               colorImage.flagImageChanged();

               // Convert to OpenCV Mat
             
        //if(gatherFrames < numFramesToGather) {
            videoFbo.readToPixels(curVideoFrames[gatherFrames % numFramesToGather]);
//            curVideoFrames[gatherFrames].setImageType(OF_IMAGE_COLOR);
        //}

        // Process mask for contour detection
//        maskFbo.readToPixels(colorImage.getPixels());
        // When reading pixels, specify the format explicitly
//        unsigned char* pixels = colorImage.getPixels().getData();
//        maskFbo.readToPixels(pixels, GL_RGB); // Match your FBO's format

        // Alternative approach using ofPixels as an intermediate
        ofPixels pixels;
        shapeMaskFbo.readToPixels(pixels);
        colorImage.setFromPixels(pixels);
        
        colorImage.flagImageChanged();

        // Convert to OpenCV Mat
        cv::Mat colorMat = cv::Mat(colorImage.getHeight(), colorImage.getWidth(), CV_8UC3, colorImage.getPixels().getData());
//        cv::Mat colorMat = cv::Mat(colorImage.getHeight(), colorImage.getWidth(), CV_8UC3, colorImage.getPixels().getData());

 
        // Extract the green channel
        std::vector<cv::Mat> channels(3);
        cv::split(colorMat, channels);
        cv::Mat greenChannel = channels[1];

        // Threshold the green channel to create a binary mask
        cv::Mat binaryMat;
        cv::threshold(greenChannel, binaryMat, 128, 255, cv::THRESH_BINARY);


        cv::Mat morphMat;
         int morphSize = 10;  // Adjust size as needed
         cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(2 * morphSize + 1, 2 * morphSize + 1));
         cv::morphologyEx(binaryMat, morphMat, cv::MORPH_CLOSE, element);

         // Convert back to ofxCvGrayscaleImage
         binaryMask.setFromPixels(morphMat.data, colorImage.getWidth(), colorImage.getHeight());


        // Convert back to ofxCvGrayscaleImage
//        binaryMask.setFromPixels(binaryMat.data, colorImage.getWidth(), colorImage.getHeight());

        // Use this mask for contour detection
        contourFinder.findContours(binaryMask, 5, (camWidth * camHeight), 200, true); // find holes
    }

    // Check if video is done
    if (video.getCurrentFrame() >= video.getTotalNumFrames() - 1 || video.getIsMovieDone()) {
        ofLog() << "Video " << currentVideoIndex << " finished. Current frame: " << video.getCurrentFrame() 
                << " Total frames: " << video.getTotalNumFrames();
        
        // Stop and close current video
        video.stop();
        video.close();
        
        // Move to next video
        currentVideoIndex = (currentVideoIndex + 1) % videoFiles.size();
        ofLog() << "Moving to video index: " << currentVideoIndex;
        
        // Reset gathering frames counter
        gatherFrames = 0;
        
        // Load and play next video
        if (videoFiles.size() > 0) {
            string nextVideoPath = videoFiles[currentVideoIndex].getAbsolutePath();
            ofLog() << "Loading next video: " << nextVideoPath;
            
            if (video.load(nextVideoPath)) {
                video.setLoopState(OF_LOOP_NONE);  // Ensure video doesn't loop
                video.play();
                video.setVolume(0);
                ofLog() << "Successfully loaded new video";
            } else {
                ofLogError() << "Failed to load video: " << nextVideoPath;
                // Try next video if this one fails
                currentVideoIndex = (currentVideoIndex + 1) % videoFiles.size();
            }
        }
    }
    #endif
}

//--------------------------------------------------------------
void ofApp::draw() {
    float w = 1920;
    float h = 1080;
//    colorImage.draw(0,0,w,h);
//    videoFbo.draw(0,0,w/2,h/2);
//    maskFbo.draw(w/2,h/2,w/2,h/2);
//
//    return;
     if(test){
         ofLog() << "TESTINGGG";
        
         int kk = ofGetFrameNum() % (numFramesToGather+60);
         kk = ofClamp(kk, 0, numFramesToGather-1);
        
         int k = kk;
          bestFits[testCharNum].videoFrames[k].setImageType(OF_IMAGE_COLOR);
         saveImgs[k].setFromPixels(bestFits[testCharNum].videoFrames[k]);
         //ofClear(0);

         // Start transformation
         ofPushMatrix();
         // Move to center of screen
         ofTranslate(ofGetWidth()/2., ofGetHeight()/2.);

         // Calculate rotation
         float rot = fmod(bestFits[testCharNum].rotation, 6.28318530718f);
         float currentRotation = ofMap(float(k), 0., numFramesToGather, 0., rot);
         ofRotateZRad(currentRotation);

         // Draw image centered on the rotation point
         saveImgs[k].draw(-bestFits[testCharNum].centerX, -bestFits[testCharNum].centerY);

         // Draw contours in the SAME coordinate system
         ofSetColor(255, 0, 0);  // Red for visibility
         ofBeginShape();
         for(int i = 0; i < bestFits[testCharNum].bodyContours.size(); i++) {
             ofVertex(bestFits[testCharNum].bodyContours[i].x - bestFits[testCharNum].centerX,
                      bestFits[testCharNum].bodyContours[i].y - bestFits[testCharNum].centerY);
         }
         ofEndShape(true);
         ofSetColor(255);  // Reset color

         ofPopMatrix();  // Match the ofPushMatrix() from earlier

         // Draw the letter
         ofPushMatrix();
         ofTranslate(250, 100, 0);
         ofBeginShape();
         for(int i = 0; i < (int)bestFits[testCharNum].charContours.size(); i++){
             ofVertex(bestFits[testCharNum].charContours[i].x,
                      bestFits[testCharNum].charContours[i].y);
         }
         ofEndShape(true);
         ofPopMatrix();  // Match the ofPushMatrix() for letter drawing

         return;
     }
//    video.draw(0, 0, w, h);
   // imgMask.draw(0, 0, w, h);
    
#ifdef USE_LIVE_VIDEO
    mClient.draw(0,0,w/6.,h/6.);

    syphonImage.setFromPixels(pixels);  // Load the pixels into the ofImage

    // Now you can draw syphonImage in your draw() method
    syphonImage.draw(w/6.,h/6.,w/6.,h/6.);  // Draw the image at position (0, 0)
# else
    videoFbo.draw(0,0,w/6.,h/6.);  // Draw the image at position (0, 0)
    shapeMaskFbo.draw(0,h/6,w/6.,h/6.);  // Draw the image at position (0, 0)
#endif
//
//    fbo.begin();
//    ofClear(0);
//
//    fbo.end();
//    fbo.draw(0,0);

    // Make sure any remaining matrix operations are properly closed
    if (contourFinder.blobs.size() > 0){
        contourFinder.blobs[0].draw(0,0);
        if(contourFinder.blobs[0].boundingRect.getWidth()<50 && contourFinder.blobs[0].boundingRect.getHeight()<50 ){
            return;
        }
        // draw over the centroid if the blob is a hole
        ofSetColor(255);
        if(contourFinder.blobs[0].hole){
            ofDrawBitmapString(std::string(ofRandom(1,5), '?'),
                               contourFinder.blobs[0].boundingRect.getCenter().x,
                               contourFinder.blobs[0].boundingRect.getCenter().y);
            
        }
        centerXTemp = contourFinder.blobs[0].boundingRect.getCenter().x;
        centerYTemp = contourFinder.blobs[0].boundingRect.getCenter().y;
    }
    gatherFrames++;
    if(video.getCurrentFrame() < numFramesToGather){
        
        return;
    }
//    fbo.readToPixels(pix);
//    characterImage.setFromPixels(pix);
    std::vector<std::vector<cv::Point>> points;
//    points = (std::vector<std::vector<cv::Point>>)contourFinder.blobs[0].pts;
    std::vector<std::vector<cv::Point>> cvBlobs;
//
    
    //make array of cv points for body
    if (contourFinder.blobs.size()>0) {
            std::vector<cv::Point> cvBlob;
        for (int j = 0 ; j< contourFinder.blobs[0].nPts; j++) {
                // Assuming we only need the x and y components of ofVec3f
                ofVec3f vec= contourFinder.blobs[0].pts[j];
                cvBlob.emplace_back(cv::Point(static_cast<int>(vec.x), static_cast<int>(vec.y)));
        
            }
            cvBlobs.push_back(cvBlob);
        // Calculate the centroid of the current blob
   
   
        }
    // make array of character points
    // matchShapes !!!!!
    for (int f=0 ; f < numChars ; f++ ){
        if(cvBlobs.size() >0 && bestFits[f].charContours.size() >0  ){
            
            if(cvBlobs[0].size() >0 ){
                float newSim = cv::matchShapes(cvBlobs[0],bestFits[f].charContours,cv::CONTOURS_MATCH_I2,0);
                if (newSim  < bestFits[f].sim){
                    bestFits[f].sim=newSim;
                   // saveImg.setFromPixels(video.getPixels());
                    
                    bestFits[f].centerX = centerXTemp;
                    bestFits[f].centerY = centerYTemp;
                    auto [resampledContour1, resampledContour2] = resampleContours(cvBlobs[0], bestFits[f].charContours);
                    rotation = calculateRotationFromContours(resampledContour1, resampledContour2);
                    homoMat = cvMatToOFMatrix( cv::estimateAffinePartial2D(resampledContour1, resampledContour2) );
                    
                    ofFill();
                    //ofSetColor(ofColor(255,0,0));
                    drawPointsOpenFrameworksStyle(resampledContour1);

                    //ofSetColor(ofColor(0,255,0));
                    drawPointsOpenFrameworksStyle(bestFits[f].charContours);
                    
                    ofNoFill();
                    bestFits[f].rotation = rotation;//atan2(homoMat(1, 0), homoMat(0, 0));;
                    bestFits[f].bodyContours =cvBlobs[0];
                    bestFits[f].charContours =resampledContour2;
                    for (int fram = 0 ; fram < numFramesToGather ; fram++){
                        
                            int index = (gatherFrames - (numFramesToGather - fram)) % numFramesToGather;
                            if (index < 0) index += numFramesToGather;  // Ensure positive index
                            
                            bestFits[f].videoFrames[fram] = curVideoFrames[index];
//                        bestFits[f].videoFrames[fram] = curVideoFrames[((gatherFrames%numFramesToGather)+fram) % numFramesToGather];
                    }
                    ofLog()<< ofToString(newSim) + " newSIM";
                    break;
                }
               
               ofDrawBitmapStringHighlight(ofToString(newSim) + " newLowsim", 12,39);
                
            }
        }
        // ofPopMatrix();
    }
        //videoFbo.draw(0,0,w/2,h/2);
        //maskFbo.draw(w/2,h/2,w/2,h/2);
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
    if (key == 's'){
        mClient.set(dir.getDescription(dir.size() - 1));
        cout << "setting syphon";
    }
    float w = 1920;
    float h = 1080;
    
    // make the actual output image
    if(key =='t'){
        testCharNum = ofRandom(numChars);
        test = !test;
    }
    if (key == 'a'){
        ofLog() << "saveProgress";
        std::string sport = "mixxx";
        
        // Define crop dimensions
        const int cropWidth = 1094;
        const int cropHeight = 974;
        
        // Helper function to normalize angle to [-PI, PI]
        auto normalizeAngle = [](float angle) {
            while (angle > PI) angle -= TWO_PI;
            while (angle < -PI) angle += TWO_PI;
            return angle;
        };
        
        // Reallocate saveImgs and img
        for(int i = 0; i < numFramesToGather; i++) {
            saveImgs[i].allocate(cropWidth, cropHeight, OF_IMAGE_COLOR);
        }
        img.allocate(cropWidth, cropHeight, OF_IMAGE_COLOR);
        
        // Create FBOs
        ofImage maskImage;
        ofFbo frameFbo;
        ofFbo cropFbo;
        maskImage.allocate(cropWidth, cropHeight, OF_IMAGE_COLOR);
        frameFbo.allocate(1920, 1080, GL_RGBA);
        cropFbo.allocate(cropWidth, cropHeight, GL_RGBA);
        
        for (int c = 0; c < numChars; c++) {
            float targetRotation = normalizeAngle(bestFits[c].rotation);
            float cropX = bestFits[c].centerX - cropWidth/2;
            float cropY = bestFits[c].centerY - cropHeight/2;
            // Calculate centers and offsets
            float screenCenterX = 1920/2;
            float screenCenterY = 1080/2;
            
            for (int k = 0; k < numFramesToGather; k++) {
                frameFbo.begin();
                ofClear(0, 0, 0, 0);
                ofPushMatrix();
                
                // Center on shape's position
                ofTranslate(bestFits[c].centerX, bestFits[c].centerY);
                
                // Apply rotation
                float progress = float(k) / float(numFramesToGather);
                float currentRotation = progress * targetRotation;
                ofRotateZRad(currentRotation);
                
                // Draw the video frame
                saveImgs[k].setFromPixels(bestFits[c].videoFrames[k]);
                saveImgs[k].draw(-bestFits[c].centerX, -bestFits[c].centerY, 1920, 1080);  // Explicitly set width and height
                
                ofPopMatrix();
                frameFbo.end();
                
                // Crop the frame
                cropFbo.begin();
                ofClear(0, 0, 0, 0);
                frameFbo.draw(-cropX, -cropY);
                cropFbo.end();
                
                // Save the frame
                ofPixels cropPixels;
                cropFbo.readToPixels(cropPixels);
                img.setFromPixels(cropPixels);
                
                std::string saveDir = sport + "/" + (bestFits[c].letter >= 'a' ? "_" : "") + ofToString(bestFits[c].letter);
                if(k == 0) makeDir(saveDir);
                img.save(saveDir + "/" + ofToString(bestFits[c].sim) + "_" + ofToString(k,4,'0') + ".png");
            }
            
            // Generate mask with identical transformation
            frameFbo.begin();
            ofClear(0);
            ofPushMatrix();
            
            // Use same transformation as final frame
            ofTranslate(bestFits[c].centerX, bestFits[c].centerY);
            ofRotateZRad(targetRotation);
            
            // Draw mask shape
            ofSetColor(255);
            ofFill();
            ofBeginShape();
            for(int i = 0; i < bestFits[c].bodyContours.size(); i++) {
                ofVertex(bestFits[c].bodyContours[i].x - bestFits[c].centerX, 
                        bestFits[c].bodyContours[i].y - bestFits[c].centerY);
            }
            ofEndShape(true);
            
            ofPopMatrix();
            frameFbo.end();
            
            // Crop mask using same transformation as frames
            cropFbo.begin();
            ofClear(0);
            frameFbo.draw(-cropX, -cropY);
            cropFbo.end();
            
            // Process mask
            cropFbo.readToPixels(maskImage.getPixels());
            maskImage.update();
            
            // Apply blur to mask
            cv::Mat maskMat = cv::Mat(maskImage.getHeight(), maskImage.getWidth(), CV_8UC3, maskImage.getPixels().getData());
            cv::Mat blurredMat;
            cv::GaussianBlur(maskMat, blurredMat, cv::Size(9, 9), 3.0);
            
            // Now add 60 frames with fading effect
            for (int fadeFrame = 0; fadeFrame < 60; fadeFrame++) {
                float fadeAmount = fadeFrame / 60.0f;
                
                // Draw the last frame with the same transformation as the mask
                frameFbo.begin();
                ofClear(0, 0, 0, 0);
                ofPushMatrix();
                
                // Use exact same transformation as mask
                ofTranslate(bestFits[c].centerX, bestFits[c].centerY);
                ofRotateZRad(targetRotation);
                
                // Draw the last frame centered like the mask
                saveImgs[numFramesToGather-1].setFromPixels(bestFits[c].videoFrames[numFramesToGather-1]);
                saveImgs[numFramesToGather-1].draw(-bestFits[c].centerX, -bestFits[c].centerY, 1920, 1080);  // Explicitly set width and height
                
                ofPopMatrix();
                frameFbo.end();
                
                // Crop using same region as before
                cropFbo.begin();
                ofClear(0);
                frameFbo.draw(-cropX, -cropY);
                cropFbo.end();
                
                ofPixels lastFramePixels;
                cropFbo.readToPixels(lastFramePixels);
                ofPixels maskPixels = maskImage.getPixels();
                ofPixels outputPixels;
                outputPixels.allocate(cropWidth, cropHeight, OF_IMAGE_COLOR);
                
                // Process each pixel with blur
                const int blurRadius = 3;
                for(int y = 0; y < cropHeight; y++) {
                    for(int x = 0; x < cropWidth; x++) {
                        // Calculate average mask value in blur radius
                        float avgMask = 0;
                        int samples = 0;
                        
                        for(int by = -blurRadius; by <= blurRadius; by++) {
                            for(int bx = -blurRadius; bx <= blurRadius; bx++) {
                                int sx = x + bx;
                                int sy = y + by;
                                
                                if(sx >= 0 && sx < cropWidth && sy >= 0 && sy < cropHeight) {
                                    avgMask += maskPixels.getColor(sx, sy).getBrightness();
                                    samples++;
                                }
                            }
                        }
                        
                        avgMask /= samples;
                        
                        ofColor pixelColor = lastFramePixels.getColor(x, y);
                        float blendAmount = ofMap(avgMask, 0, 255, 0, 1, true);
                        
                        // Blend between darkening and lightening based on blur
                        float darkAmount = 1.0 - (0.8 * fadeAmount);
                        float lightAmount = 1.0 + (0.2 * fadeAmount);
                        float finalAmount = ofLerp(darkAmount, lightAmount, blendAmount);
                        
                        pixelColor.setBrightness(min(255.0f, pixelColor.getBrightness() * finalAmount));
                        outputPixels.setColor(x, y, pixelColor);
                    }
                }
                
                img.setFromPixels(outputPixels);
                
                std::string saveDir = sport + "/" + (bestFits[c].letter >= 'a' ? "_" : "") + ofToString(bestFits[c].letter);
                img.save(saveDir + "/" + ofToString(bestFits[c].sim) + "_" + ofToString(numFramesToGather + fadeFrame,4,'0') + ".png");
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}
