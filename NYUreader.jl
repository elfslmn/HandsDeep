using FileIO, Images, MAT
include("util.jl");

searchdir(path,key) = filter(x->contains(x,key), readdir(path));
searchlabel(path,key) = filter(x->contains(x,key), readdir(path));
cubeSize = 128;

function readNYUTesting()
   dir = Pkg.dir(pwd(),"data","NYU", "test");

   # read test images
   filenames = searchdir(dir, "png");
   xtst = Array{Float32, 4}(cubeSize,cubeSize,1,length(filenames));

   info("Reading NYU test sequence ...")
   info("Reading test images...")
   for i in 1:(length(filenames))
      path = joinpath(dir,filenames[i]);
      raw = rawview(channelview(load(path)));
      dpt = map(Int16, ((2^8).*raw[2,:,:] .+ raw[3,:,:]));
      #preprocess the image, extract hand,normalize depth to [-1,1]
      p = preprocess(dpt, getNYUCameraParameters(), cubeSize; dset=1);
      img = reshape(convert(Array{Float32,2}, p), cubeSize,cubeSize,1,1);
      xtst[:,:,:,i] = img;
   end
   info("xtst:", summary(xtst))

   info("Reading test labels...")
   file = matopen(joinpath(dir,"joint_data.mat"))
   label = read(file, "joint_uvd")[1,:,:,:]
   close(file)
   ytst = Array{Float32, 2}(size(label,2)*size(label,3), size(label,1));
   for i in 1:size(label,1) #each image
       for j in 1:size(label,2) #each joint
           ytst[3*(j-1)+1,i] = label[i,j,1];
           ytst[3*(j-1)+2,i] = label[i,j,2];
           ytst[3*(j-1)+3,i] = label[i,j,3];
       end
   end
   ytst = ytst ./1000;
   info("ytst:", summary(ytst))

   return (xtst, ytst)
end

#sz: early stop size
function readNYUTraining(;sz = -1)
    dir = Pkg.dir(pwd(),"data","NYU", "train");

    # read test images
    filenames = searchdir(dir, "png");
    xtrn = Array{Float32, 4}(cubeSize,cubeSize,1,length(filenames));

    info("Reading NYU training sequence ...")
    info("Reading training images...")
    for i in 1:(length(filenames))
       path = joinpath(dir,filenames[i]);
       raw = rawview(channelview(load(path)));
       dpt = map(Int16, ((2^8).*raw[2,:,:] .+ raw[3,:,:]));
       #preprocess the image, extract hand,normalize depth to [-1,1]
       p = preprocess(dpt, getNYUCameraParameters(), cubeSize; dset=1);
       img = reshape(convert(Array{Float32,2}, p), cubeSize,cubeSize,1,1);
       xtrn[:,:,:,i] = img;
    end
    info("xtrn:", summary(xtrn))

    info("Reading training labels...")
    file = matopen(joinpath(dir,"joint_data.mat"))
    label = read(file, "joint_uvd")[1,:,:,:]
    close(file)
    label = label[1:size(filenames,1),:,:];
    ytrn = Array{Float32, 2}(size(label,2)*size(label,3), size(label,1));
    for i in 1:size(label,1) #each image
        for j in 1:size(label,2) #each joint
            ytrn[3*(j-1)+1,i] = label[i,j,1];
            ytrn[3*(j-1)+2,i] = label[i,j,2];
            ytrn[3*(j-1)+3,i] = label[i,j,3];
        end
    end
    ytrn = ytrn ./1000;
    info("ytrn:", summary(ytrn))

    return (xtrn, ytrn)
end

#= return a 4-elements tuple
:param fx: focal length in x direction
:param fy: focal length in y direction
:param ux: principal point in x direction
:param uy: principal point in y direction =#
function getNYUCameraParameters()
    return (588.03, 587.07, 320., 240.)
end
