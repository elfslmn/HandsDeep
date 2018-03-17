using FileIO

function readICVLTesting()
current = pwd();
icvl_test_images_dir = current * "/ICVL/Testing/Depth/" ;
# read test images
tst1 = readdir(icvl_test_images_dir*"test_seq_1");
tst2 = readdir(icvl_test_images_dir*"test_seq_2");
xtst = Array{Float32, 4}(240,320,1,(size(tst1,1)+size(tst2,1)));

info("Reading ICVL test sequence 1...")
for i in 1:size(tst1,1)
path = icvl_test_images_dir*"test_seq_1/"*tst1[i];
img = reshape(convert(Array{Float32,2}, load(path)), 240,320,1,1);
xtst[:,:,:,i] = img;
end

info("Reading ICVL test sequence 2...")
for j in 1:size(tst2,1)
path = icvl_test_images_dir*"test_seq_2/"*tst2[j];
img = reshape(convert(Array{Float32,2}, load(path)), 240,320,1,1);
xtst[:,:,:,(j+size(tst1,1))] = img;
end
info("xtst:", summary(xtst))

# read test labels
l1 = open(readdlm, "ICVL/Testing/test_seq_1.txt")[:,2:end];
l2 = open(readdlm, "ICVL/Testing/test_seq_2.txt")[:,2:end];
ytst = vcat(l1,l2);
info("ytst", summary(ytst))
return (xtst, ytst)
end



function readICVLTraining()
current = pwd();
icvl_training_images_dir = current * "/ICVL/Training/Depth/" ;

# read training images
xtrn = Array{Float32, 4}(240,320,1,1);
folders = readdir(icvl_training_images_dir);
for j in 1:3  #size(folders,1)
f = folders[j]
info("Reading training folder ", f)
files = readdir(icvl_training_images_dir * f);
x = Array{Float32, 4}(240,320,1,size(files,1));
for i in 1:size(files,1)
path = icvl_training_images_dir * f*"/"* files[i];
img = reshape(convert(Array{Float32,2}, load(path)), 240,320,1,1);
x[:,:,:,i] = img;
end
if size(xtrn,1)> 1
xtrn = cat(4, xtrn, x);
else
xtrn = x;
end
end

info("xtrn:", summary(xtrn))

# read training labels
ytrn = open(readdlm, "ICVL/Training/labels.txt")[1:size(xtrn,4),2:end];

return (xtrn, ytrn)
end
