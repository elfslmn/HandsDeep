using FileIO, Images

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
   xtst = Array{Float32, 4}(240,320,1,(size(tst1,1)+size(tst2,1)));

   info("Reading ICVL test sequence 1...")
   for i in 1:size(tst1,1)
      path = joinpath(dir,"test_seq_1", tst1[i]);
     # TODO preprocess the image, extract hand,normalize depth to [-1,1]
      #img = imresize(load(path), 128,128);
      #img = reshape(convert(Array{Float32,2}, img ), 128,128,1,1);
      img = reshape(convert(Array{Float32,2}, load(path) ), 240,320,1,1);
      xtst[:,:,:,i] = img;
   end

   info("Reading ICVL test sequence 2...")
   for j in 1:size(tst2,1)
      path = joinpath(dir,"test_seq_2", tst2[j]);
      # TODO preprocess the image, extract hand,normalize depth to [-1,1]
      #img = imresize(load(path), 128,128);
      #img = reshape(convert(Array{Float32,2}, img), 128,128,1,1);
      img = reshape(convert(Array{Float32,2}, load(path) ), 240,320,1,1);
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
   info("ytst", summary(ytst))
   return (xtst, ytst)
end


function readICVLTraining(;s::Int64=-1) #!!!!!!!!Not reading all of them memory overflow
   dir = Pkg.dir(pwd(),"data","ICVL", "Training", "Depth");

   # read training images
   #xtrn = Array{Float32, 4}(128,128,1,1);
   xtrn = Array{Float32, 4}(240,320,1,1);
   ytrn = Array{Float32, 2}(48,1);
   whole = open(readdlm, "data/ICVL/Training/labels.txt");
   foldernames = readdir(dir);
   if s < 0
       s = size(folders,1);
   end

   for j in 1:s
      folder = foldernames[j]
      info("Reading training folder ", folder)
      files = searchdir(joinpath(dir,folder), "png");
      #x = Array{Float32, 4}(128,128,1,size(files,1));
      x = Array{Float32, 4}(240,320,1,size(files,1));
      #read images
      for i in 1:size(files,1)
         path = joinpath(dir, f, files[i]);
          # TODO preprocess the image, extract hand,normalize depth to [-1,1]
         #img = imresize(load(path), 128,128);
         #img = reshape(convert(Array{Float32,2}, img), 128,128,1,1);
          img = reshape(convert(Array{Float32,2}, load(path) ), 240,320,1,1);
         x[:,:,:,i] = img;
      end
      #read labels
      part = whole[contains.(whole[:,1],folder*"/image_"), :];
      y = part[.!contains.(part[:,1],("/"*folder)), :];

      # There is some images that has no labels. This is to ingore those
      if size(y,1) > size(files,1)
          info("Label matrix will truncate for folder ", folder)
          y = y[1:size(files,1)];
      elseif size(y,1) < size(files,1)
          info("İmage matrix will truncate for folder ", folder)
          x = x[:,:,:,1:size(y,1)];
      end

      if size(xtrn,4)> 1
         xtrn = cat(4, xtrn, x);
         ytrn = cat(2, ytrn, y[:,2:end]');
      else
         xtrn = x;
         ytrn = y[:,2:end]';
      end
   end
   ytrn = convert(Array{Float32,2}, ytrn);
   info("xtrn:", summary(xtrn))
   info("ytrn:", summary(ytrn))

   return (xtrn, ytrn)
end

#= return a 4-elements tuple
:param fx: focal length in x direction
:param fy: focal length in y direction
:param ux: principal point in x direction
:param uy: principal point in y direction =#
function getICVLCameraParameters()
    return (241.42, 241.42, 160., 120.)
end
