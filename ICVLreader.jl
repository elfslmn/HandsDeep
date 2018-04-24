using FileIO, Images;
include("util.jl");
#= ICVL Label descriptions:
Each line is corresponding to one image.
Each line has 16x3 numbers, which indicates (x, y, z) of 16 joint locations.
Note that (x, y) are in pixels and z is in mm.
The order of 16 joints is Palm, Thumb root, Thumb mid, Thumb tip, Index root, Index mid, Index tip,
Middle root, Middle mid, Middle tip, Ring root, Ring mid, Ring tip, Pinky root, Pinky mid, Pinky tip.=#

searchdir(path,key) = filter(x->contains(x,key), readdir(path));
searchlabel(path,key) = filter(x->contains(x,key), readdir(path));
imgSize = 128;

# sz is for early stop
function readICVLTesting(;sz = -1)
    dir = Pkg.dir(pwd(),"data","ICVL", "Testing", "Depth");
    l1 = open(readdlm, "data/ICVL/Testing/test_seq_1.txt");
    l2 = open(readdlm, "data/ICVL/Testing/test_seq_2.txt");
    files = vcat(l1,l2);

    xtst = Array{Float32, 4}(imgSize,imgSize,1,size(files,1));
    ytst = Array{Float32, 2}(48,size(files,1));

    c = 0;
    param = getICVLCameraParameters();
    info("Starting to read testing set...")
    for i in 1:size(files,1)
         path = joinpath(dir,files[i,1]);
         if(!isfile(path))
            continue;
         end
         # preprocess the image, extract hand,normalize depth to [-1,1]
         c +=1;
         p, com = preprocess(convert(Array{Float32,2}, load(path)), getICVLCameraParameters(), imgSize)
         img = reshape(p, imgSize,imgSize,1,1);

         # ground truth
         joints = files[i,2:end]';
         joints3D = jointsImgTo3D(joints, param);
         com3D = jointImgTo3D(com, param);
         joints3DCrop = copy(joints3D);
         for i in 1:16
             joints3DCrop[3*(i-1)+1] -= com3D[1];
             joints3DCrop[3*(i-1)+2] -= com3D[2];
             joints3DCrop[3*(i-1)+3] -= com3D[3];
         end

         xtst[:,:,:,c] = img;
         ytst[:,c] = joints3DCrop./250; # cropped, normalized, 3D

         if c == sz
             break;
         end
    end
    xtst = xtst[:,:,:,1:c];
    ytst = ytst[:,1:c];

    info("xtst:", summary(xtst))
    info("ytst:", summary(ytst))

    return (xtst, ytst)
end

# sz is for early stop
function readICVLTraining(;sz = -1)
   dir = Pkg.dir(pwd(),"data","ICVL", "Training", "Depth");

   # read training images
   if isfile(joinpath(pwd(),"icvl_trnfiles.txt"))
       files = open(readdlm, "icvl_trnfiles.txt");
   else
       whole = open(readdlm, "data/ICVL/Training/labels.txt");
       files = removeNotUsedLabels(whole);
       whole = nothing
       open("icvl_trnfiles.txt", "w") do io
            writedlm(io, files);
       end
   end

   xtrn = Array{Float32, 4}(imgSize,imgSize,1,size(files,1));
   ytrn = Array{Float32, 2}(48,size(files,1));

   c = 0;
   param = getICVLCameraParameters();
   info("Starting to read training set...")
   for i in 1:size(files,1)
        path = joinpath(dir,files[i,1]);
        if(!isfile(path))
           continue;
        end
        # preprocess the image, extract hand,normalize depth to [-1,1]
        c +=1;
        p, com = preprocess(convert(Array{Float32,2}, load(path)), getICVLCameraParameters(), imgSize)
        img = reshape(p, imgSize,imgSize,1,1);

        # ground truth
        joints = files[i,2:end]';
        joints3D = jointsImgTo3D(joints, param);
        com3D = jointImgTo3D(com, param);
        joints3DCrop = copy(joints3D);
        for i in 1:16
            joints3DCrop[3*(i-1)+1] -= com3D[1];
            joints3DCrop[3*(i-1)+2] -= com3D[2];
            joints3DCrop[3*(i-1)+3] -= com3D[3];
        end

        xtrn[:,:,:,c] = img;
        ytrn[:,c] = joints3DCrop./250; # cropped, normalized, 3D

        if c%1000 == 0
            info(c," images are read..");
        end

        if c == sz
            break;
        end
   end
   xtrn = xtrn[:,:,:,1:c];
   ytrn = ytrn[:,1:c];

   info("xtrn:", summary(xtrn))
   info("ytrn:", summary(ytrn))

   return (xtrn, ytrn)
end

function removeNotUsedLabels(whole)
    part = whole[.!contains.(whole[:,1],"112-5/"), :];
    part = part[.!contains.(part[:,1],"67-5/"), :];
    part = part[.!contains.(part[:,1],"157-5/"), :];
    part = part[.!contains.(part[:,1],"180/"), :];
    part = part[.!contains.(part[:,1],"22-5/"), :];
    part = part[.!contains.(part[:,1],"45/"), :];
    part = part[.!contains.(part[:,1],"90/"), :];
    part = part[.!contains.(part[:,1],"135/"), :];
    return part;
end

#= return a 4-elements tuple
:param fx: focal length in x direction
:param fy: focal length in y direction
:param ux: principal point in x direction
:param uy: principal point in y direction =#
function getICVLCameraParameters()
    return (241.42, 241.42, 160., 120.)
end
