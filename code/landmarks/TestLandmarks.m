%% load resized images to make a test sample

impath_start = 'C:\Users\saj3\Documents\cycleGAN\099000_resized\0';
impath_end = '.png';

images_resized = zeros(304,304,3,1000);
for i = 1:1000
    idx = i + 99000-1;
    str = strcat(impath_start,num2str(idx),impath_end);
    image_resized = imread(str);
    images_resized(:,:,:,i) = image_resized;
    if i == 500
        fprintf('half way through...')
    end
end

%% test sample
im_idx = 919; % e.g. '18' for '099018.png'
marks = [114 143 190 143 151 183 115 220 193 220];


%% plot results

figure()
imshow(uint8(images_resized(:,:,:,im_idx+1))), hold on
for n = 1:2:10
  scatter(marks(n),marks(n+1),'MarkerEdgeColor',[1 1 1], 'MarkerFaceColor',[n/10 n/10 n/10])
end
