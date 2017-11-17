function [panorama] = local_stitch(img1, img2, Hdlt, X, Y, width, height, offset)


    warp_img1 = uint8(zeros([height width 3]));
    warp_img2 = uint8(zeros([height width 3]));
    
    warp_img1(offset(2):(offset(2)+size(img1,1)-1),offset(1):(offset(1)+size(img1,2)-1),:) = img1;
    
    for xidx=1:width
        for yidx=1:height
            x_grididx = min(find(X>xidx,1),size(X,2));
            y_grididx = min(find(Y>yidx,1),size(Y,1));
            
            grididx = x_grididx*size(Y,1)+y_grididx;
            
            H = Hdlt{grididx};
            
            
        end
    end

end