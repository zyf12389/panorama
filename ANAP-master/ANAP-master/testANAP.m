run('C:/Users/ww/Desktop/panorama/code/cvpr14_SPHP_code/vlfeat-0.9.14/toolbox/vl_setup')
addpath('mexfiles');
addpath('modelspecific');

img0=imread('Stitch_Dataset/street_01.jpg');
img1 = imread('Stitch_Dataset/temple_1.jpg');
img2 = imread('Stitch_Dataset/temple_2.jpg');

%% Feature detection using SIFT and matching


global fitfn resfn degenfn 
fitfn = 'homography_fit';
resfn = 'homography_res';
degenfn = 'homography_degen';


[f1, d1] = vl_sift(single(rgb2gray(img1))) ;
% [f2, d2] = vl_sift(single(rgb2gray(img2)),'PeakThresh', 0,'edgethresh',500) ;
[f2, d2] = vl_sift(single(rgb2gray(img2))) ;


[match_idxs, scores] = vl_ubcmatch(d1, d2) ;
f1 = f1(:,match_idxs(1,:));
f2 = f2(:,match_idxs(2,:));
[~, score_idx] = sort(scores);

%% Remove outlier using RANSAC
M     = 100;  % Number of iteration for RANSAC.

inliers = ransac(f1,f2,M);

%%  Global homography
figure;subplot(1,2,1);imshow(img1);hold on;
scatter(f1(1,:),f1(2,:),'r');scatter(f1(1,inliers),f1(2,inliers),'g');
subplot(1,2,2);imshow(img2);hold on;
scatter(f1(1,:),f1(2,:),'r');scatter(f1(1,inliers),f1(2,inliers),'g');

[H, A] = global_homography(f1(:,inliers), f2(:,inliers));

%% Image Stitching
[panorama, panorama_width, panorama_height, offset] = image_stitch(img1,img2,H);
figure; imshow(panorama);

p1 = f1(:,inliers);
p2 = f2(:,inliers);
tmp_p2 = H\[p2(1:2,:);ones(1,size(p2,2))];
p2(1,:) = tmp_p2(1,:)./tmp_p2(3,:);
p2(2,:) = tmp_p2(2,:)./tmp_p2(3,:);
hold on;
scatter(p1(1,:),p1(2,:),'go');
scatter(p2(1,:),p2(2,:),'ro');

%% Local Homography
width_grid = 100;
height_grid = 100;
[PjsX, PjsY] = meshgrid(linspace(1,panorama_width,width_grid), linspace(1,panorama_height,height_grid));

Pjs = [PjsX(:)-offset(1), PjsY(:)-offset(2)]';

Hl = local_homography(f1(:,inliers), A, Pjs);
disp(Hl(:,:));
%% Local Stitching

[panorama] = imagewarping(double(panorama_height),double(panorama_width), double(img1), Hl,double(offset), PjsX(1,:), PjsY(:,1)');
panorama = reshape(uint8(panorama),size(panorama,1),size(panorama,2)/3,3);
figure;
imshow(panorama);