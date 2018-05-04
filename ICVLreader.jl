using FileIO, Images;
#include("util.jl");

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
function readICVLTesting(;sz = -1,raw = false)
    dir = Pkg.dir(pwd(),"data","ICVL", "Testing", "Depth");
    l1 = open(readdlm, "data/ICVL/Testing/test_seq_1.txt");
    l2 = open(readdlm, "data/ICVL/Testing/test_seq_2.txt");
    files = vcat(l1,l2);

    xtst = Array{Float32, 4}(imgSize,imgSize,1,size(files,1));
    ytst = Array{Float32, 2}(48,size(files,1));
    coms3D = Array{Float32, 2}(3,size(files,1)); # center of masses in world coor.
    trMats = Array{Float32, 3}(3,3,size(files,1)); # transformation matrices
    if raw
        ximg = Array{Float32, 3}(240,320,size(files,1));
    end

    c = 0;
    param = getICVLCameraParameters();
    info("Starting to read test set...")
    for i in 1:size(files,1)
         path = joinpath(dir,files[i,1]);
         if(!isfile(path))
            continue;
         end
         # preprocess the image, extract hand,normalize depth to [-1,1]
         dpt = convert(Array{Float32,2}, load(path));
         p, com, M = preprocess(dpt, getICVLCameraParameters(), imgSize)
         if any(isnan, p)
             print("skip ", i);
             continue;
         end

         c +=1;
         if raw
             ximg[:,:,c] = dpt;
         end
         img = reshape(p, imgSize,imgSize,1,1);
         # ground truth
         joints = files[i,2:end]';
         joints3D = jointsImgTo3D(joints, param);
         com3D = jointImgTo3D(com, param);
         joints3DCrop = copy(joints3D);
         for j in 1:16
             joints3DCrop[3*(j-1)+1] -= com3D[1];
             joints3DCrop[3*(j-1)+2] -= com3D[2];
             joints3DCrop[3*(j-1)+3] -= com3D[3];
         end

         xtst[:,:,:,c] = img;
         ytst[:,c] = joints3DCrop./250; # cropped, normalized, 3D
         coms3D[:,c] = com3D;
         trMats[:,:,c] = M;

         if c == sz
             break;
         end
    end
    xtst = xtst[:,:,:,1:c];
    ytst = ytst[:,1:c];
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
        return (xtst, ytst,coms3D, trMats )
    end


end

# sz is for early stop
function readICVLTraining(;sz = -1, raw = false)
   dir = Pkg.dir(pwd(),"data","ICVL", "Training", "Depth");

   # read training images
   if isfile(joinpath(pwd(),"data/ICVL/Training/icvl_trnfiles.txt"))
       files = open(readdlm, "data/ICVL/Training/icvl_trnfiles.txt");
   else
       whole = open(readdlm, "data/ICVL/Training/labels.txt");
       files = removeNotUsedLabels(whole);
       whole = nothing
       open("data/ICVL/Training/icvl_trnfiles.txt", "w") do io
            writedlm(io, files);
       end
   end

   xtrn = Array{Float32, 4}(imgSize,imgSize,1,size(files,1));
   ytrn = Array{Float32, 2}(48,size(files,1));
   coms3D = Array{Float32, 2}(3,size(files,1)); # center of masses in world coor.
   trMats = Array{Float32, 3}(3,3,size(files,1)); # transformation matrices
   if raw
       ximg = Array{Float32, 3}(240,320,size(files,1));
   end

   c = 0;
   param = getICVLCameraParameters();
   info("Starting to read training set...")
   for i in 1:size(files,1)
        path = joinpath(dir,files[i,1]);
        if(!isfile(path))
           continue;
        end
        # preprocess the image, extract hand,normalize depth to [-1,1]
        dpt = convert(Array{Float32,2}, load(path));
        p, com , M = preprocess(dpt, param, imgSize)
        if any(isnan, p)
            println("skip ", i);
            continue;
        end
        c +=1;
        if raw
            ximg[:,:,c] = dpt;
        end

        img = reshape(p, imgSize,imgSize,1,1);
        # ground truth
        joints = files[i,2:end]';
        joints3D = jointsImgTo3D(joints, param);
        com3D = jointImgTo3D(com, param);

        joints3DCrop = copy(joints3D);
        for j in 1:16
            joints3DCrop[3*(j-1)+1] -= com3D[1];
            joints3DCrop[3*(j-1)+2] -= com3D[2];
            joints3DCrop[3*(j-1)+3] -= com3D[3];
        end

        xtrn[:,:,:,c] = img;
        ytrn[:,c] = joints3DCrop./250; # cropped, normalized, 3D
        coms3D[:,c] = com3D;
        trMats[:,:,c] = M;

        if c%1000 == 0
            info(c," images are read..");
        end

        if c == sz
            break;
        end
   end
   xtrn = xtrn[:,:,:,1:c];
   ytrn = ytrn[:,1:c];
   coms3D = coms3D[:,1:c];
   trMats = trMats[:,:,1:c];
   if raw
       ximg = ximg[:,:,1:c];
   end

   info("xtrn:", summary(xtrn))
   info("ytrn:", summary(ytrn))

   if raw
       info("Raw images:", summary(ximg))
       return (xtrn, ytrn, coms3D, trMats, ximg);
   else
       return (xtrn, ytrn, coms3D, trMats);
   end

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

function getICVLJointImages(;sz=-1)
    dir = Pkg.dir(pwd(),"data","ICVL", "Training", "Depth");

    # read training images
    if isfile(joinpath(pwd(),"data/ICVL/Training/icvl_trnfiles.txt"))
        files = open(readdlm, "data/ICVL/Training/icvl_trnfiles.txt");
    else
        whole = open(readdlm, "data/ICVL/Training/labels.txt");
        files = removeNotUsedLabels(whole);
        whole = nothing
        open("data/ICVL/Training/icvl_trnfiles.txt", "w") do io
             writedlm(io, files);
        end
    end

    njoint = 16;

    #(height, width, channel, image count, joint count);
    x64 = Array{Float32, 5}(64,64,1,size(files,1),njoint);
    y = Array{Float32, 3}(3,size(files,1),njoint);

    info("Starting to read training set...")
    c=0;
    for i in 1:size(files,1)
         path = joinpath(dir,files[i,1]);
         if(!isfile(path)) || i==17855
            continue;
         end
         img = convert(Array{Float32,2}, load(path));

         # ground truth
         joints = files[i,2:end];
         for j in 0:njoint-1
             p = extractPatch(img, joints[3*j+1:3*j+3], 64);
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

function showRes(i,w)
    dir = Pkg.dir(pwd(),"data","ICVL", "Testing", "Depth");
    l1 = open(readdlm, "data/ICVL/Testing/test_seq_1.txt");
    l2 = open(readdlm, "data/ICVL/Testing/test_seq_2.txt");
    files = vcat(l1,l2);

    path = joinpath(dir,files[i,1]);
    img = convert(Array{Float32,2}, load(path));
    p, com , M = preprocess(img, getICVLCameraParameters(), 128)
    joints = files[i,2:end];
    com3D = jointImgTo3D(com, getICVLCameraParameters());

    pred = embedNet(w,reshape(p,128,128,1,1); drop=false);
    pred2D = convertCrop3DToImg(com3D,pred,0)
    #showAnnotation(img, joints, pred = (pred2D+joints)./2);
    showAnnotation(img, joints, pred = pred2D)
end
