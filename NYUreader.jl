using FileIO, Images, MAT
include("util.jl");

searchdir(path,key) = filter(x->contains(x,key), readdir(path));
searchlabel(path,key) = filter(x->contains(x,key), readdir(path));
imgSize = 128;

function readNYUTesting(;sz=-1, raw = false)
   dir = Pkg.dir(pwd(),"data","NYU", "test");

   # read test images
   filenames = searchdir(dir, "png");
   xtst = Array{Float32, 4}(imgSize,imgSize,1,length(filenames));

   info("Reading NYU test sequence ...")
   file = matopen(joinpath(dir,"joint_data.mat"))
   joints3D = read(file, "joint_xyz")[1,:,:,:]
   close(file)

   joints3DCrop = Array{Float32, 1}(size(joints3D,2)*size(joints3D,3));
   ytst = Array{Float32, 2}(size(joints3D,2)*size(joints3D,3), size(joints3D,1));
   coms3D = Array{Float32, 2}(3,length(filenames)); # center of masses in world coor.
   trMats = Array{Float32, 3}(3,3,length(filenames)); # transformation matrices
   validIndices= Array{Int64}(length(filenames));
   if raw
       ximg = Array{Int16, 3}(480,640,length(filenames));
   end

   param = getNYUCameraParameters();
   c = 0;
   info("Starting to read images...")
   for i in 1:(length(filenames))
      path = joinpath(dir,filenames[i]);
      rawimg = rawview(channelview(load(path)));
      dpt = map(Int16, ((2^8).*rawimg[2,:,:] .+ rawimg[3,:,:]));
      #preprocess the image, extract hand,normalize depth to [-1,1]
      (p, com, M) = preprocess(dpt, getNYUCameraParameters(), imgSize; dset=1);
      if any(isnan, com) || std(p) < 0.5
          info(:NotInclude , i , filenames[i]);
          continue;
      end
      c += 1;
      xtst[:,:,:,c] = reshape(convert(Array{Float32,2}, p), imgSize,imgSize,1,1);

      com3D = jointImgTo3D(map(Float32,com), param);
      for j in 1:size(joints3D,2) #each joint
          joints3DCrop[3*(j-1)+1] = joints3D[i,j,1] - com3D[1];
          joints3DCrop[3*(j-1)+2] = joints3D[i,j,2] - com3D[2];
          joints3DCrop[3*(j-1)+3] = joints3D[i,j,3] - com3D[3];
      end
      ytst[:,c]= joints3DCrop./300;
      coms3D[:,c] = com3D;
      trMats[:,:,c] = M;
      validIndices[c]=i;
      if raw
          ximg[:,:,c] = dpt;
      end

      if c%1000 == 0
          info(c," images are read..");
      end

      if c == sz
          break;
      end
   end
   xtst = xtst[:,:,:,1:c]
   ytst = ytst[:,1:c]
   coms3D = coms3D[:,1:c];
   trMats = trMats[:,:,1:c];
   validIndices = validIndices[1:c];
   if raw
       ximg = ximg[:,:,1:c];
   end

   info("xtst:", summary(xtst))
   info("ytst:", summary(ytst))

   if raw
       info("Raw images:", summary(ximg))
       return (xtst, ytst,coms3D, trMats, validIndices, ximg )
   else
       return (xtst, ytst,coms3D, trMats, validIndices)
   end
end

#sz: early stop size
function readNYUTraining(;sz=-1, raw = false)
   dir = Pkg.dir(pwd(),"data","NYU", "train");

   # read test images
   filenames = searchdir(dir, "png");
   xtrn = Array{Float32, 4}(imgSize,imgSize,1,length(filenames));

   info("Reading NYU training sequence ...")
   file = matopen(joinpath(dir,"joint_data.mat"))
   joints3D = read(file, "joint_xyz")[1,:,:,:]
   close(file)

   joints3DCrop = Array{Float32, 1}(size(joints3D,2)*size(joints3D,3));
   ytrn = Array{Float32, 2}(size(joints3D,2)*size(joints3D,3), size(joints3D,1));
   coms3D = Array{Float32, 2}(3,length(filenames)); # center of masses in world coor.
   trMats = Array{Float32, 3}(3,3,length(filenames)); # transformation matrices
   validIndices= Array{Int64}(length(filenames));
   if raw
       ximg = Array{Int16, 3}(480,640,length(filenames));
   end

   param = getNYUCameraParameters();
   c = 0;
   info("Starting to read images...")
   for i in 1:(length(filenames))
      path = joinpath(dir,filenames[i]);
      rawimg = rawview(channelview(load(path)));
      dpt = map(Int16, ((2^8).*rawimg[2,:,:] .+ rawimg[3,:,:]));
      #preprocess the image, extract hand,normalize depth to [-1,1]
      (p, com, M) = preprocess(dpt, getNYUCameraParameters(), imgSize; dset=1);
      if any(isnan, com) || std(p) < 0.5
          info(:NotInclude , i , filenames[i]);
          continue;
      end
      c += 1;
      xtrn[:,:,:,c] = reshape(convert(Array{Float32,2}, p), imgSize,imgSize,1,1);

      com3D = jointImgTo3D(map(Float32,com), param);
      for j in 1:size(joints3D,2) #each joint
          joints3DCrop[3*(j-1)+1] = joints3D[i,j,1] - com3D[1];
          joints3DCrop[3*(j-1)+2] = joints3D[i,j,2] - com3D[2];
          joints3DCrop[3*(j-1)+3] = joints3D[i,j,3] - com3D[3];
      end
      ytrn[:,c]= joints3DCrop./300;
      coms3D[:,c] = com3D;
      trMats[:,:,c] = M;
      validIndices[c]=i;

      if raw
          ximg[:,:,c] = dpt;
      end

      if c%5000 == 0
          info(c," images are read..");
      end

      if c == sz
          break;
      end
   end
   xtrn = xtrn[:,:,:,1:c]
   ytrn = ytrn[:,1:c]
   coms3D = coms3D[:,1:c];
   trMats = trMats[:,:,1:c];
   validIndices = validIndices[1:c];
   if raw
       ximg = ximg[:,:,1:c];
   end

   info("xtrn:", summary(xtrn))
   info("ytrn:", summary(ytrn))

   if raw
       info("Raw images:", summary(ximg))
       return (xtrn, ytrn, coms3D, trMats,validIndices, ximg);
   else
       return (xtrn, ytrn, coms3D, validIndices, trMats);
   end
end

#= return a 4-elements tuple
:param fx: focal length in x direction
:param fy: focal length in y direction
:param ux: principal point in x direction
:param uy: principal point in y direction =#
function getNYUCameraParameters()
    return (588.03, 587.07, 320., 240.)
end
