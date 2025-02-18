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


	// parameters for the neural network
	float downsampleRatio = 0.25f;
	float videoWidth = 1920;
	float videoHeight = 1080;
	float batchSize = 1.0f;
	float numChannels = 3.0f;


	imgMask.allocate(videoWidth, videoHeight, OF_IMAGE_GRAYSCALE);
    grayImage.allocate(videoWidth, videoHeight);
    colorImage.allocate(videoWidth, videoHeight);
    characterImage.allocate(videoWidth, videoHeight);
    fbo.allocate(videoWidth, videoHeight);
    pixels.allocate(1920,1080, OF_IMAGE_COLOR);
//    syphonPixels.allocate(1920,1080, OF_IMAGE_COLOR_ALPHA);
//    syphonImage.allocate(1920,1080, OF_IMAGE_COLOR);
	imgOut.allocate(videoWidth,videoHeight, OF_IMAGE_COLOR_ALPHA);

    
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
            }
        
        
        // this is for
        
        // A 0
        // a 32
        // Q 16
        // q 48
        // d 35
        // g 38
        
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
//        else{
            
            bestFits[i].charContours = cvBlob2; // one big
        //}
        
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
//        video.load("in_sam2/swim00037.mp4");
//     videoMask.load("sam2/swim00037.mp4");
    
    video.load("in_sam2/shot_999.mp4");
     videoMask.load("sam2/shot_999.mp4");
//    videoMask.setFrameRate(video.getFrameRate());
    ////
            video.play();
    videoMask.play();
//            video.setLoopState(OF_LOOP_NONE);
        #endif

    //using Syphon app Simple Server, found at http://syphon.v002.info/
  //  mClient.set("","OBS");
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
    if (video.getCurrentFrame() >= video.getTotalNumFrames() - 1) {
        video.setFrame(video.getTotalNumFrames() - 1);
        videoMask.setFrame(videoMask.getTotalNumFrames() - 1);
    }
    video.update();
//    int frameIdx = video.getCurrentFrame(); // Get current frame of the video
//     videoMask.setFrame(frameIdx);
    videoMask.update();
    if(video.isFrameNew() && videoMask.isFrameNew() && (test==false)) {
        // Force videoMask to match
       
        
        video.getPixels();
        
        fbo.begin();
        video.draw(0,0,ofGetWidth(), ofGetHeight() );
        fbo.end();
        
        fbo.readToPixels(pixels);
        
#endif
            fbo.begin();
            
            
#ifdef USE_LIVE_VIDEO
            
            mClient.draw(0,0,ofGetWidth(), ofGetHeight() );
            
            
#else
            video.draw(0,0,ofGetWidth(), ofGetHeight());
            
            
#endif
            
            fbo.end();
            
            fbo.readToPixels(curVideoFrames[gatherFrames%numFramesToGather]);
            curVideoFrames[gatherFrames%numFramesToGather].setImageType(OF_IMAGE_COLOR);
        
  
        colorImage.setFromPixels(videoMask.getPixels()); // Load the mask image

        // Convert to OpenCV Mat
        cv::Mat colorMat = cv::Mat(colorImage.getHeight(), colorImage.getWidth(), CV_8UC3, colorImage.getPixels().getData());

        // Extract the green channel
        std::vector<cv::Mat> channels(3);
        cv::split(colorMat, channels);
        cv::Mat greenChannel = channels[1];

        // Threshold the green channel to create a binary mask
        cv::Mat binaryMat;
        cv::threshold(greenChannel, binaryMat, 128, 255, cv::THRESH_BINARY);

        // Convert back to ofxCvGrayscaleImage
        binaryMask.setFromPixels(binaryMat.data, colorImage.getWidth(), colorImage.getHeight());

        // Use this mask for contour detection
        contourFinder.findContours(binaryMask, 5, (camWidth * camHeight), 200, true); // find holes

#ifndef USE_LIVE_VIDEO
    }
#endif


}

//--------------------------------------------------------------
void ofApp::draw() {
    float w = 1920;
    float h = 1080;

    if(test) {
        // Get frame index, clamped to valid range
        int frameIndex = ofClamp(ofGetFrameNum() % (numFramesToGather+60), 0, numFramesToGather-1);
        
        // Load the saved frame
        saveImgs[frameIndex].setFromPixels(bestFits[testCharNum].videoFrames[frameIndex]);
        
        ofClear(0);
        ofPushMatrix();
        
        // Center the transformation
        float centerX = ofGetWidth()/2.;
        float centerY = ofGetHeight()/2.;
        ofTranslate(centerX, centerY);
        
        // Calculate and apply rotation
        float totalRotation = bestFits[testCharNum].rotation;
        float currentRotation = ofMap(frameIndex, 0, numFramesToGather-1, 0, totalRotation);
        ofRotateZDeg(currentRotation);
        
        // Draw the frame centered on its saved center point
        saveImgs[frameIndex].draw(
            -bestFits[testCharNum].centerX, 
            -bestFits[testCharNum].centerY
        );
        
        // Draw contours for debugging
        ofNoFill();
        ofSetColor(255, 0, 0);
        drawPointsOpenFrameworksStyle(bestFits[testCharNum].bodyContours);
        ofSetColor(255);
        
        ofPopMatrix();
        
        // Draw reference character
        ofPushMatrix();
        ofTranslate(250, 100);
        ofBeginShape();
        drawPointsOpenFrameworksStyle(bestFits[testCharNum].charContours);
        ofEndShape(true);
        ofPopMatrix();
        
        return;
    }

    // ... existing video/mask drawing code ...

    // Handle frame gathering
    if (contourFinder.blobs.size() > 0 && 
        contourFinder.blobs[0].boundingRect.getWidth() >= 50 && 
        contourFinder.blobs[0].boundingRect.getHeight() >= 50) {
        
        // Save center point
        centerXTemp = contourFinder.blobs[0].boundingRect.getCenter().x;
        centerYTemp = contourFinder.blobs[0].boundingRect.getCenter().y;
        
        // Save current frame
        int currentFrameIndex = gatherFrames % numFramesToGather;
        curVideoFrames[currentFrameIndex].setFromPixels(pixels);
        curVideoFrames[currentFrameIndex].setImageType(OF_IMAGE_COLOR);
        
        gatherFrames++;
    }

    if(gatherFrames < numFramesToGather) {
        return;
    }

    // Process gathered frames for shape matching
    if (contourFinder.blobs.size() > 0) {
        std::vector<cv::Point> currentBlob;
        // Convert blob points to CV points
        for (const auto& pt : contourFinder.blobs[0].pts) {
            currentBlob.emplace_back(cv::Point(pt.x, pt.y));
        }

        // Find best matching character
        for (int charIndex = 0; charIndex < numChars; charIndex++) {
            if (bestFits[charIndex].charContours.empty()) continue;

            float similarity = cv::matchShapes(currentBlob, 
                                            bestFits[charIndex].charContours,
                                            cv::CONTOURS_MATCH_I2, 0);

            if (similarity < bestFits[charIndex].sim) {
                // Save match data
                bestFits[charIndex].sim = similarity;
                bestFits[charIndex].centerX = centerXTemp;
                bestFits[charIndex].centerY = centerYTemp;
                
                // Calculate rotation and save contours
                auto [resampledBody, resampledChar] = resampleContours(currentBlob, 
                                                                      bestFits[charIndex].charContours);
                bestFits[charIndex].rotation = calculateRotationFromContours(resampledBody, resampledChar);
                bestFits[charIndex].bodyContours = resampledBody;
                bestFits[charIndex].charContours = resampledChar;

                // Save frames in correct order
                for (int frameIdx = 0; frameIdx < numFramesToGather; frameIdx++) {
                    int sourceIdx = (gatherFrames - numFramesToGather + frameIdx) % numFramesToGather;
                    if (sourceIdx < 0) sourceIdx += numFramesToGather;
                    bestFits[charIndex].videoFrames[frameIdx] = curVideoFrames[sourceIdx];
                }
            }
        }
    }
}
