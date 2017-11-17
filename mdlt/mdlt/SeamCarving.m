function  overlap  = SeamCarving(I,I2)

%   Detailed explanation goes here
%calculating image gradient as an energy image
I_copy=I;
[rmax, cmax,tmp] = size(I);
img = double(I)/255;
img = rgb2gray(img);

img2 = double(I2)/255;
img2 = rgb2gray(img2);

[Ix,Iy]=gradient(img);

[Ix2,Iy2]=gradient(img2);

Gimg=Ix+Iy;
Gimg=abs(Gimg);

Gimg2=Ix2+Iy2;
Gimg2=abs(Gimg2);

I_diff1=I(:,:,1)-I2(:,:,1);
I_diff2=I(:,:,2)-I2(:,:,2);
I_diff3=I(:,:,3)-I2(:,:,3);
I_diff=I_diff1+I_diff2+I_diff3;
I_diff=I_diff.^2;
Gimg=Gimg-Gimg2;
Gimg=Gimg.^2;
Gimg=Gimg+im2double(I_diff);
figure;
imshow(Gimg);
title('test');
%calculating sum of pixels value in each column
test2=zeros(rmax,cmax);
for row = 1 :rmax
    for col = 1:cmax
        if row == 1
            test2(row,col)=Gimg(row,col);
        end
        if row>1
            if col ==1
                tmp=[test2(row-1,col),test2(row-1,col+1)];
                test2(row,col)= Gimg(row,col)+min(tmp);
            end
            if col>1 && col<cmax
                tmp1=[test2(row-1,col),test2(row-1,col+1),test2(row-1,col-1)];
                test2(row,col)= Gimg(row,col)+min(tmp1);
            end
            if col == cmax
                tmp2=[test2(row-1,col),test2(row-1,col-1)];
                test2(row,col)= Gimg(row,col)+min(tmp2);
            end
        end
    end
end
minval=min(test2(rmax,:));
locations=find(test2(rmax,:)==minval);
[x,y]=size(locations);
%back traking to find the seam
index_result=zeros(rmax,2);
for loc=1:y
    j = locations(1,loc);
    for row=rmax:-1:1
        if row==rmax
            I(row,j,1)=255;
            I(row,j,2)=0;
            I(row,j,3)=0;
            index_result(row,1)=row;
            index_result(row,2)=j;
        end
        if row < rmax
            if j==1
                tmp=[test2(row+1,j),test2(row+1,j+1)];
                [C,index]=min(tmp);
                if index==1
                    I(row+1,j,1)=255;
                    I(row+1,j,2)=0;
                    I(row+1,j,3)=0;
                    index_result(row+1,1)=row+1;
                    index_result(row+1,2)=j;
                end
                if index==2
                    I(row+1,j+1,1)=255;
                    I(row+1,j+1,2)=0;
                    I(row+1,j+1,3)=0;
                    index_result(row+1,1)=row+1;
                    index_result(row+1,2)=j+1;
                    j=j+1;
                end
            end
            if j>1 && j<cmax
                tmp1=[test2(row+1,j),test2(row+1,j+1),test2(row+1,j-1)];
                [C,index]=min(tmp1);
                if index==1
                    I(row+1,j,1)=255;
                    I(row+1,j,2)=0;
                    I(row+1,j,3)=0;
                    index_result(row+1,1)=row+1;
                    index_result(row+1,2)=j;
                end
                if index==2
                    I(row+1,j+1,1)=255;
                    I(row+1,j+1,2)=0;
                    I(row+1,j+1,3)=0;
                    index_result(row+1,1)=row+1;
                    index_result(row+1,2)=j+1;
                    j=j+1;
                end
                if index==3
                    I(row+1,j-1,1)=255;
                    I(row+1,j-1,2)=0;
                    I(row+1,j-1,3)=0;
                    index_result(row+1,1)=row+1;
                    index_result(row+1,2)=j-1;
                    j=j-1;
                end
            end
            if j == cmax
                tmp2=[test2(row+1,j),test2(row+1,j-1)];
                [C,index]=min(tmp2);
                if index==1
                    I(row+1,j,1)=255;
                    I(row+1,j,2)=0;
                    I(row+1,j,3)=0;
                    index_result(row+1,1)=row+1;
                    index_result(row+1,2)=j;
                end
                if index==2
                    I(row+1,j-1,1)=255;
                    I(row+1,j-1,2)=0;
                    I(row+1,j-1,3)=0;
                    index_result(row+1,1)=row+1;
                    index_result(row+1,2)=j-1;
                    j=j-1;
                end
            end
        end
    end
end
overlap=I_copy;
for i=1:rmax
    for j=1:cmax
    if index_result(i,2)<=j
        overlap(i,j,1)=I2(i,j,1);
        overlap(i,j,2)=I2(i,j,2);
        overlap(i,j,3)=I2(i,j,3);
    end
    end
end
figure;
imshow(I);
title('seam');
figure;
imshow(overlap);
end