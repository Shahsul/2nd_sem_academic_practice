clear all;close all;clc;

% Read a video
video = VideoReader('test_video.mp4','CurrentTime',0.50);
counter=0;
while hasFrame(video)
    frames = readFrame(video);
    counter=counter+1;
    
    % Create a cascade detector object.
    faceDetector = vision.CascadeObjectDetector();
    
    %bound the detected positions using a bounding box
    bbox = step(faceDetector,rgb2gray(frames)); 

    % Draw the returned bounding box around the detected face.
    detected_face = insertShape(frames, 'Rectangle', bbox);
    
    
    %crop the face only
    face= imcrop(detected_face,[bbox(1,:)+2]);
    
    %extract the eye features
     %Right Eye
     re=vision.CascadeObjectDetector('RightEye','MergeThreshold',40);
     bbox1=step(re,face);
     dre=insertShape(face,'Rectangle',bbox1);
 
     %Left Eye
     le=vision.CascadeObjectDetector('LeftEye','MergeThreshold',40);
     bbox2=step(le,dre);
     dle=insertShape(dre,'Rectangle',bbox2);
   
        
    baseFileName = sprintf('sample %d.png', counter);
    fullFileName = fullfile('C:','Users','acer','Documents','MATLAB','Thesis_practice','Extracted images', baseFileName);
    imwrite(dre, fullFileName);
    if counter==1100
        figure; imshow(detected_face);
        
%detect feature points to track the fac
        feature_points = detectMinEigenFeatures(rgb2gray(frames),'ROI', bbox);

% Display the detected points.
        figure, imshow(detected_face), hold on, title('Detected features');
        plot(feature_points);
    end
end
