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

function readICVLTesting()
   dir = Pkg.dir(pwd(),"data","ICVL", "Testing", "Depth");

   # read test images
   tst1 = searchdir(joinpath(dir,"test_seq_1"), "png");
   tst2 = searchdir(joinpath(dir,"test_seq_2"), "png");
   #xtst = Array{Float32, 4}(128,128,1,(size(tst1,1)+size(tst2,1)));
   xtst = Array{Float32, 4}(128,128,1,(size(tst1,1)+size(tst2,1)));

   info("Reading ICVL test sequence 1...")
   for i in 1:size(tst1,1)
      path = joinpath(dir,"test_seq_1", tst1[i]);
     #preprocess the image, extract hand,normalize depth to [-1,1]
      p = preprocess(convert(Array{Float32,2}, load(path)), getICVLCameraParameters())
      img = reshape(p, 128,128,1,1);
      #img = reshape(convert(Array{Float32,2}, load(path) ), 240,320,1,1);
      xtst[:,:,:,i] = img;
   end

   info("Reading ICVL test sequence 2...")
   for j in 1:size(tst2,1)
      path = joinpath(dir,"test_seq_2", tst2[j]);
      # TODO preprocess the image, extract hand,normalize depth to [-1,1]
      p = preprocess(convert(Array{Float32,2}, load(path)), getICVLCameraParameters())
      img = reshape(p, 128,128,1,1);
      #img = reshape(convert(Array{Float32,2}, load(path) ), 240,320,1,1);
      xtst[:,:,:,(j+size(tst1,1))] = img;
   end
   info("xtst:", summary(xtst))

   # read test labels
   l1 = open(readdlm, "data/ICVL/Testing/test_seq_1.txt")[:,2:end];
   l2 = open(readdlm, "data/ICVL/Testing/test_seq_2.txt")[:,2:end];
   ytst = vcat(l1,l2);
   ytst = ytst';
   # TODO normalize depth to [-1,1] ??
   ytst = convert(Array{Float32,2},ytst);
   info("ytst: ", summary(ytst))
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

   xtrn = Array{Float32, 4}(128,128,1,size(files,1));
   ytrn = Array{Float32, 2}(48,size(files,1));

   c = 0;
   info("Starting to read training set...")
   for i in 1:size(files,1)
        path = joinpath(dir,files[i,1]);
        if(!isfile(path))
           continue;
        end
        # preprocess the image, extract hand,normalize depth to [-1,1]
        c +=1;
        p = preprocess(convert(Array{Float32,2}, load(path)), getICVLCameraParameters())
        img = reshape(p, 128,128,1,1);
        joints = files[i,2:end]';
        #img = reshape(convert(Array{Float32,2}, load(path) ), 240,320,1,1);
        xtrn[:,:,:,c] = img;
        ytrn[:,c] = joints;

        if c%1000 == 0
            info(c," images are read..");
        end

        if c == sz
            break;
        end
   end
   xtrn = xtrn[:,:,:,1:c];
   ytrn = ytrn[:,1:c];

   #ytrn = convert(Array{Float32,2}, ytrn); ???? bu niye vardı
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
