%% 
%use imageDataStore to get all images in a folder
path = fullfile("C:\Users\nursa\OneDrive\DrinkCont\drink\drink box\");
imds = imageDatastore(path,"IncludeSubfolders",true,LabelSource="foldernames");
%to get number of files
n = numel(imds.Files);

%%
%data augmentation
for i = 1:n
    img = readimage(imds,i);
    %resize images for 

    wrtpath=fullfile("containerx3/drink box/");
    name=sprintf("%sd%d.jpg",wrtpath,i);
    imwrite(img,name,"jpg");

    %rotate the images between -90 degree to 90 degree
    rotate=randomAffine2d(Rotation=[-90 90]);
    rot=imwarp(img,rotate,OutputView=affineOutputView(size(img),rotate));
    wrtpath=fullfile("containerx3/drink box/");
    name=sprintf("%sdRo%d.jpg",wrtpath,i);
    imwrite(rot,name,"jpg");

    %scale
    scale=randomAffine2d(Scale=[1,2]);
    sc=imwarp(img,scale,OutputView=affineOutputView(size(img),scale));
    wrtpath=fullfile("containerx3/drink box/");
    name=sprintf("%sdS%d.jpg",wrtpath,i);
    imwrite(sc,name,"jpg");
 
    %reflect
    reflect=randomAffine2d(XReflection=true,YReflection=true);
    ref=imwarp(img,reflect,OutputView=affineOutputView(size(img),reflect));
    wrtpath=fullfile("containerx3/drink box/");
    name=sprintf("%sdRe%d.jpg",wrtpath,i);
    imwrite(ref,name,"jpg");

end
