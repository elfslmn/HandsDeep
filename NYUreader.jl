using FileIO, Images, MAT
include("util.jl");

searchdir(path,key) = filter(x->contains(x,key), readdir(path));
searchlabel(path,key) = filter(x->contains(x,key), readdir(path));
imgSize = 128;

function readNYUTesting(;sz=-1)
   dir = Pkg.dir(pwd(),"data","NYU", "test");

   # read test images
   filenames = searchdir(dir, "png");
   xtst = Array{Float32, 4}(imgSize,imgSize,1,length(filenames));

   info("Reading NYU test sequence ...")
   file = matopen(joinpath(dir,"joint_data.mat"))
   label = read(file, "joint_xyz")[1,:,:,:]
   close(file)

   joints3DCrop = Array{Float32, 1}(size(label,2)*size(label,3));
   ytst = Array{Float32, 2}(size(label,2)*size(label,3), size(label,1));
   param = getNYUCameraParameters();

   c = 0;
   for i in 1:(length(filenames))
      path = joinpath(dir,filenames[i]);
      raw = rawview(channelview(load(path)));
      dpt = map(Int16, ((2^8).*raw[2,:,:] .+ raw[3,:,:]));
      #preprocess the image, extract hand,normalize depth to [-1,1]
      (p, com) = preprocess(dpt, getNYUCameraParameters(), imgSize; dset=1);
      if any(isnan, com) || std(p) < 0.5
          info(:NotInclude , i , filenames[i]);
          continue;
      end

      img = reshape(convert(Array{Float32,2}, p), imgSize,imgSize,1,1);
      xtst[:,:,:,i] = img;

      com3D = jointImgTo3D(map(Float32,com), param);
      for j in 1:size(label,2) #each joint
          joints3DCrop[3*(j-1)+1] = label[i,j,1] - com3D[1];
          joints3DCrop[3*(j-1)+2] = label[i,j,2] - com3D[2];
          joints3DCrop[3*(j-1)+3] = label[i,j,3] - com3D[3];
      end
      ytst[:,i]= joints3DCrop./300;

      c += 1;
      if c == sz
          break;
      end
   end
   xtst = xtst[:,:,:,1:c]
   ytst = ytst[:,1:c]

   info("xtst:", summary(xtst))
   info("ytst:", summary(ytst))

   return (xtst, ytst)
end

#sz: early stop size
function readNYUTraining(;sz = -1)
    dir = Pkg.dir(pwd(),"data","NYU", "train");

    # read train images
    filenames = searchdir(dir, "png");
    xtrn = Array{Float32, 4}(imgSize,imgSize,1,length(filenames));

    info("Reading NYU training sequence ...")
    file = matopen(joinpath(dir,"joint_data.mat"))
    label = read(file, "joint_xyz")[1,:,:,:]
    close(file)

    joints3DCrop = Array{Float32, 1}(size(label,2)*size(label,3));
    ytrn = Array{Float32, 2}(size(label,2)*size(label,3), size(label,1));
    param = getNYUCameraParameters();

    c = 0;
    for i in 1:(length(filenames))
       path = joinpath(dir,filenames[i]);
       raw = rawview(channelview(load(path)));
       dpt = map(Int16, ((2^8).*raw[2,:,:] .+ raw[3,:,:]));
       #preprocess the image, extract hand,normalize depth to [-1,1]
       (p, com) = preprocess(dpt, getNYUCameraParameters(), imgSize; dset=1);
       if any(isnan, com) || std(p) < 0.5
           info(:NotInclude , i , filenames[i]);
           continue;
       end

       img = reshape(convert(Array{Float32,2}, p), imgSize,imgSize,1,1);
       xtrn[:,:,:,i] = img;

       com3D = jointImgTo3D(map(Float32,com), param);
       for j in 1:size(label,2) #each joint
           joints3DCrop[3*(j-1)+1] = label[i,j,1] - com3D[1];
           joints3DCrop[3*(j-1)+2] = label[i,j,2] - com3D[2];
           joints3DCrop[3*(j-1)+3] = label[i,j,3] - com3D[3];
       end
       ytrn[:,i]= joints3DCrop./300;

       c += 1;
       if c == sz
           break;
       end
    end
    xtrn = xtrn[:,:,:,1:c]
    ytrn = ytrn[:,1:c]

    info("xtrn:", summary(xtrn))
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
