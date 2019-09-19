%I = imread('G:\kinect_head_pose\kinect_head_pose_db\cut\01\frame_00020_rgb.png');

%I = imread('G:\HeadPoseImageDatabase\Person13\person13100-90+0.jpg')
%I = imread('G:\kinect_head_pose\kinect_head_pose_db\hpdb\03\frame_00410_rgb.png');%266
I = imread('D:\matlabcode\04.jpg');
%I = imread('D:\matlabcode\video_1\100.jpg');
%I = imread('G:\DrivFace\DrivImages\DrivImages\20130529_01_Driv_030_lr.jpg')
%I = imread('G:\DrivFace\DrivImages\DrivImages\20130529_01_Driv_032_lr.jpg')
% I = imread('G:\DrivFace\DrivImages\DrivImages\20130529_02_Driv_099_f .jpg')
%I = imread('G:\DrivFace\DrivImages\DrivImages\20130529_02_Driv_142_lr.jpg')
% I = imread('G:\DrivFace\DrivImages\DrivImages\20130530_03_Driv_055_f .jpg')
%I = imread('G:\DrivFace\DrivImages\DrivImages\20130530_04_Driv_031_f .jpg')%4 l
%I = imread('G:\DrivFace\DrivImages\DrivImages\20130530_04_Driv_087_lr.jpg')%4 r
%I = imread('G:\DrivFace\DrivImages\DrivImages\20130530_04_Driv_018_ll.jpg')%4
%I = imresize(I,[224 224]);
%pretrained = load('head_pose_detector_all.mat');
%pretrained = load('faster_rcnn_detector.mat');
pretrained = load('FDDB_5_detector.mat');
detector = pretrained.detector;
[bboxes,scores] = detect(detector,I);

% I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
% imshow(I)


bboxes
x1 = bboxes(1);
y1 = bboxes(2);
w = bboxes(3);
h = bboxes(4);


cc = imcrop(I,[x1 y1 w h]);
x_c = x1 +w/2
y_c = y1+h/2
% 
cc = imresize(cc,[32 32]);
%imshow(cc)
cc = double(cc)/255;
% % I(:,:,:,1) = I
% I = reshape(I,32,32,3,1)
% size(I)
% 
load('head_pose_estimation.mat')
%load('prima_estimator.mat')
a = predict(net,cc)
b = [a(2) a(1) a(3)]
% eul = [0 pi/2 0];
rotmZYX = eul2rotm(b)
mi_ro = rotmZYX'
x_axis = rotmZYX*[100 0 0]'
y_axis = rotmZYX*[0 100 0]';
z_axis = rotmZYX*[0 0 100]';
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
% subplot(1,3,1)
imshow(I);
line([x_c x_c+x_axis(2)], [y_c y_c-x_axis(3)],'LineWidth',4,'Color','red')
line([x_c x_c+y_axis(2)], [y_c y_c-y_axis(3)],'LineWidth',4,'Color','green')
line([x_c x_c+z_axis(2)], [y_c y_c-z_axis(3)],'LineWidth',4,'Color','blue')

% imwrite(I,'D:\matlabcode\video\2.jpg','jpg')
% 
% lighting_direction = mi_ro*[1 0 0]';
% size(lighting_direction)
% l_x = lighting_direction(1);
% l_y = lighting_direction(2);
% l_z = lighting_direction(3);
% normal = load('sn.mat');
% load('albedo.mat');
% albedo = a;
% px = albedo.*normal.px;
% py = albedo.*normal.py;
% pz = albedo.*normal.pz;
% size(px)
% normal_vector = cat(3,px,py,pz);
% 
% normal_vector(:,:,3);
% size(normal_vector)
% %lighting_direction = [0 0 1]
% lighting_direction = [lighting_direction(2) lighting_direction(3) lighting_direction(1)]
% image = normal_vector(:,:,1)*lighting_direction(1)+normal_vector(:,:,2)*lighting_direction(2)+normal_vector(:,:,3)*lighting_direction(3);
% % image = px+py+pz;
% image = uint8(round(image));
% %subplot(1,3,2)
% %imshow(image);
% normal_map = cat(3,normal.px,normal.py,normal.pz);
% normal_map = (normal_map+1)*255/2;
% normal_map(:,:,1);
% normal_map = uint8(normal_map);
% %subplot(1,3,3)
% %imshow(normal_map);


