clc 
clear
params.pix_shift_x = 1;
params.pix_shift_y = 1;
params.thresh_bin = 4; %4
params.gauss_sigm = 3;


data_file = 'datasets/data_14Sep_2';
start_frame = 40;
params.min_rad = 20;
params.max_rad = 50;
params.half_window = 1;

%{
data_file = 'datasets/data_30Aug_2';
start_frame = 8;
params.min_rad = 30;
params.max_rad = 80;
params.half_window = 2;
%}

figure('Name','base image | segmentation')
imageList = dir(strcat(data_file,'/*.jpg'));
init_flag = 0;
for i=start_frame: size(imageList,1)
    img_base = imread(strcat(data_file,'/',imageList(i).name) );
     % select the ultrasound image area
    if i == start_frame
        disp("select us image area");
        [J,rect1] = imcrop(img_base);
        params.cut_xmin = rect1(2)+10;
        params.cut_ymin = rect1(1)+10;
        params.cut_height = rect1(4)-10;
        params.cut_width = rect1(3)-10;
    end
    % select the start point
    if init_flag == 0
        disp("select initial vessle area")
        [J,rect2] = imcrop(img_base);
        start_point = [rect2(2)+rect2(4)/2,rect2(1)+rect2(3)/2];
        params.start_point = start_point;
        init_flag = 1;
    end
    img_out = shift_filter_tejas(img_base,params);
    
    [m,n]=size(img_out);
    % circle detection
    [circle, edge_points] = circle_detection_wanwen_v2(img_out,params,'circle');
    if ~isempty(circle)
        img_out(round(circle.xc),round(circle.yc)) = 255; % centroid
        for i = 1:size(edge_points,1)
            img_out(round(edge_points(i,1)),round(edge_points(i,2))) = 100; % detected points in img for fitting
        end
        for theta = 0:0.01:2*pi % fitted circle
            a = round(circle.xc+circle.rad*cos(theta));
            b = round(circle.yc+circle.rad*sin(theta));
            if a > 0 && a<m && b>0 && b<n
                img_out(a,b) = 255;
            end
        end
        params.start_point = [circle.xc,circle.yc];
    else
        init_flag = 0; % select new start point
    end
    
    %{
    [ellipse, edge_points] = circle_detection_wanwen_v2(img_out,params.start_point,'ellipse');
    if ~isempty(ellipse)
        for i = 1:size(edge_points,1)
            img_out(round(edge_points(i,1)),round(edge_points(i,2))) = 100;
        end
        if 0<ellipse.xc && ellipse.xc<m && ellipse.yc>0 && ellipse.yc<n
            img_out(round(ellipse.xc),round(ellipse.yc)) = 255;
        end
        for theta = 0:0.01:2*pi
            x = ellipse.xc+ellipse.xa*cos(theta);
            y = ellipse.yc+ellipse.ya*sin(theta);
            a = round(x*cos(ellipse.tilt)+y*sin(ellipse.tilt));
            b = round(x*sin(ellipse.tilt)-y*cos(ellipse.tilt));
            if a > 0 && a<m && b>0 && b<n
                img_out(a,b) = 255;
            end
        end
        params.start_point = [ellipse.xc,ellipse.yc];
    else
        init_flag = 0;
    end
    %}
    
    montage({img_base,img_out});
    waitforbuttonpress;                                           
end
