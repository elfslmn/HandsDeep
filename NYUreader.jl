using FileIO, Images, MAT
include("util.jl");

searchdir(path,key) = filter(x->contains(x,key), readdir(path));
searchlabel(path,key) = filter(x->contains(x,key), readdir(path));
imgSize = 128;

function readNYUTesting(;sz=-1, raw = false)
   dir = Pkg.dir(pwd(),"data","NYU", "test");
   # read test images
   filenames = searchdir(dir, "png");
   if sz == -1
       sz =  length(filenames)
   end
   xtst = Array{Float32, 4}(imgSize,imgSize,1,sz);

   info("Reading NYU test sequence ...")
   file = matopen(joinpath(dir,"joint_data.mat"))
   joints3D = read(file, "joint_xyz")[1,:,:,:]
   coms2D = read(file, "joint_uvd")[1,:,33,:]
   close(file)

   jointIndices = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32] .+1;
   joints3DCrop = Array{Float32, 1}(length(jointIndices)*3);
   ytst = Array{Float32, 2}(length(jointIndices)*3, size(joints3D,1));
   coms3D = Array{Float32, 2}(3,sz); # center of masses in world coor.
   trMats = Array{Float32, 3}(3,3,sz); # transformation matrices

   if raw
       ximg = Array{Int16, 3}(480,640,sz);
   end

   param = getNYUCameraParameters();
   c = 0;
   info("Starting to read images...")
   for i in 1:(length(filenames))
      path = joinpath(dir,filenames[i]);
      rawimg = rawview(channelview(load(path)));
      dpt = map(Int16, ((2^8).*rawimg[2,:,:] .+ rawimg[3,:,:]));
      #preprocess the image, extract hand,normalize depth to [-1,1]
      (p, com, M) = preprocessNYUGtCom(dpt, param, imgSize, coms2D[i,:])
      c += 1;
      xtst[:,:,:,c] = reshape(convert(Array{Float32,2}, p), imgSize,imgSize,1,1);

      com3D = jointImgTo3D(map(Float32,com), param);
      k=0;
      for j in 1:size(joints3D,2) #each joint
          if j in jointIndices
              joints3DCrop[3*k+1] = joints3D[i,j,1] - com3D[1];
              joints3DCrop[3*k+2] = joints3D[i,j,2] - com3D[2];
              joints3DCrop[3*k+3] = joints3D[i,j,3] - com3D[3];
              k +=1;
          end
      end
      ytst[:,c]= joints3DCrop./300;
      coms3D[:,c] = com3D;
      trMats[:,:,c] = M;
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
   if raw
       ximg = ximg[:,:,1:c];
   end

   info("xtst:", summary(xtst))
   info("ytst:", summary(ytst))

   if raw
       info("Raw images:", summary(ximg))
       return (xtst, ytst,coms3D, trMats, ximg )
   else
       return (xtst, ytst,coms3D, trMats)
   end
end

#sz: early stop size
function readNYUTraining(;sz=-1, raw = false)
   dir = Pkg.dir(pwd(),"data","NYU", "train");
   # read test images
   filenames = searchdir(dir, "png");
   if sz == -1
       sz = length(filenames)
   end
   xtrn = Array{Float32, 4}(imgSize,imgSize,1,sz);

   info("Reading NYU training sequence ...")
   file = matopen(joinpath(dir,"joint_data.mat"))
   joints3D = read(file, "joint_xyz")[1,:,:,:]
   coms2D = read(file, "joint_uvd")[1,:,33,:]
   close(file)

   jointIndices = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32] .+1;
   joints3DCrop = Array{Float32, 1}(length(jointIndices)*3);
   ytrn = Array{Float32, 2}(length(jointIndices)*3, size(joints3D,1));
   coms3D = Array{Float32, 2}(3,sz); # center of masses in world coor.
   trMats = Array{Float32, 3}(3,3,sz); # transformation matrices
   if raw
       ximg = Array{Int16, 3}(480,640,sz);
   end

   param = getNYUCameraParameters();
   c = 0;
   info("Starting to read images...")
   for i in 1:(length(filenames))
      path = joinpath(dir,filenames[i]);
      rawimg = rawview(channelview(load(path)));
      dpt = map(Int16, ((2^8).*rawimg[2,:,:] .+ rawimg[3,:,:]));
      #preprocess the image, extract hand,normalize depth to [-1,1]
      (p, com, M) = preprocessNYUGtCom(dpt, param, imgSize, coms2D[i,:])
      c += 1;
      xtrn[:,:,:,c] = reshape(convert(Array{Float32,2}, p), imgSize,imgSize,1,1);

      com3D = jointImgTo3D(map(Float32,com), param);
      k=0;
      for j in 1:size(joints3D,2) #each joint
          if j in jointIndices
              joints3DCrop[3*k+1] = joints3D[i,j,1] - com3D[1];
              joints3DCrop[3*k+2] = joints3D[i,j,2] - com3D[2];
              joints3DCrop[3*k+3] = joints3D[i,j,3] - com3D[3];
              k +=1;
          end
      end
      ytrn[:,c]= joints3DCrop./300;
      coms3D[:,c] = com3D;
      trMats[:,:,c] = M;

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
   if raw
       ximg = ximg[:,:,1:c];
   end

   info("xtrn:", summary(xtrn))
   info("ytrn:", summary(ytrn))

   if raw
       info("Raw images:", summary(ximg))
       return (xtrn, ytrn, coms3D, trMats,ximg);
   else
       return (xtrn, ytrn, coms3D, trMats);
   end
end

#= return a 4-elements tuple
:param fx: focal length in x direction
:param fy: focal length in y direction
:param ux: principal point in x direction
:param uy: principal point in y direction =#
function getNYUCameraParameters()
    return (588.03, -587.07, 320., 240.)
end

function showHandNYU(i; tr=true)
    if tr
        dir= Pkg.dir(pwd(),"data","NYU", "train");
    else
        dir= Pkg.dir(pwd(),"data","NYU", "test");
    end
    filenames = searchdir(dir, "png");
    path = joinpath(dir,filenames[i]);
    rawimg = rawview(channelview(load(path)));
    dpt = map(Int16, ((2^8).*rawimg[2,:,:] .+ rawimg[3,:,:]));
    (p, com, M) = preprocess(dpt, getNYUCameraParameters(), 128; dset=1);
    imshow(p)
    imshow(dpt);
    println(std(p))
end

function getNYUJointImages(;sz=-1)
    dir = Pkg.dir(pwd(),"data","NYU", "train");
    # read test images
    filenames = searchdir(dir, "png");
    if sz == -1
        sz = length(filenames)
    end
    xtrn = Array{Float32, 4}(imgSize,imgSize,1,sz);

    info("Reading NYU training sequence ...")
    file = matopen(joinpath(dir,"joint_data.mat"))
    joints3D = read(file, "joint_xyz")[1,:,:,:]
    coms2D = read(file, "joint_uvd")[1,:,33,:]
    close(file)

    jointIndices = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32];
    param = getNYUCameraParameters();

    njoint = 14;

    #(height, width, channel, image count, joint count);
    x64 = Array{Float32, 5}(64,64,1,size(files,1),njoint);
    y = Array{Float32, 3}(3,size(files,1),njoint);

    info("Starting to read training set...")
    c=0;
    for i in 1:size(files,1)
        path = joinpath(dir,filenames[i]);
        rawimg = rawview(channelview(load(path)));
        dpt = map(Int16, ((2^8).*rawimg[2,:,:] .+ rawimg[3,:,:]));
         img = convert(Array{Float32,2}, dpt);

         # ground truth

         for j in jointIndices
             p = extractPatch(img, joints3D[3*j+1:3*j+3], 64);
             x64[:,:,:,i,j+1] = p;
         end
         c += 1;
         if c%1000 == 0
             info(c," images are read..");
         end
         if c == sz
             break;
         end
    end
    x64 = x64[:,:,:,1:c,:]
    info("Images:" , summary(x64));
    return x64;
end
