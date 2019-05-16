% load images to matrix
clc,clear,close
impath_start = 'C:\Users\saj3\Documents\cycleGAN\099000\0';
impath_end = '.png';


images = zeros(304,304,3,1000);
for i = 1:1000
    idx = i + 99000-1;
    str = strcat(impath_start,num2str(idx),impath_end);
    image = imresize(imread(str), [304, 304]);
    imwrite(image, strcat('C:\Users\saj3\Documents\cycleGAN\099000_resized\0', num2str(idx), impath_end));
    images(:,:,:,i) = image;
    if i == 500
        fprintf('half way through...')
    end
end

%% get landmarks manually
start_sample = 901;
end_sample = 1000;
marks = zeros(10, 1000);
marks(1:4,:) = repmat([114; 143; 190; 143], 1, 1000);
for i = start_sample:end_sample
    figure(1)
    imshow(uint8(images(:,:,:,i)))
    points = ginput(3);
    marks(5:end,i) = uint8(reshape(points',[1,6]));
    close
    i + 99000 - 1
end


%% plot results
for i = start_sample:end_sample
    figure()
    imshow(uint8(images(:,:,:,i))), hold on
    for n = 1:2:10
    scatter(marks(n,i),marks(n+1,i),'MarkerEdgeColor',[1-n/10 1-n/10 1-n/10], 'MarkerFaceColor',[n/10 n/10 n/10])
    end
end
%%
%marks(5:10,563) = [143; 180; 117; 223; 185; 225];
imshow(uint8(images(:,:,:,942))), hold on
for n = 1:2:10
    scatter(marks(n,942),marks(n+1,942),'MarkerEdgeColor',[1-n/10 1-n/10 1-n/10], 'MarkerFaceColor',[n/10 n/10 n/10])
end
%% make txt
fid = fopen('landmarks1000.txt', 'wt');
for i = start_sample:end_sample
    impath_start = '\099000_resized\0';
    impath_end = '.png';
    im_idx = i + 99000-1;
    str = strcat(impath_start, num2str(im_idx), impath_end);
    fprintf(fid, '%s ', str);
    for n = 1:10
        fprintf(fid, '%s ' , num2str(marks(n,i)));
    end
    fprintf(fid, '\n');
end
fclose(fid);


