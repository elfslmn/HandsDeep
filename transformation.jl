#= Normalize sample to metric 3D
:sample: joints in (x,y,z) with x,y in image coordinates and z in mm
:param: camera intirinsic parameters
:return: normalized joints in mm
=#
function jointImgTo3D(sample, param)
    ret = Array{Float32}(3);
    ret[1] = (sample[1]-param[3])*sample[3]/param[1];
    ret[2] = (sample[2]-param[4])*sample[3]/param[2];
    ret[3] = sample[3];
    return ret
end

function jointsImgTo3D(y, param)
    if length(y)%3 != 0
        println("Error: Y should be a multiple of 3");
        return;
    end
    y2 = Array{Float32}(length(y));
    for i = 1:3:length(y)
        y2[i:i+2] = jointImgTo3D(y[i:i+2], param);
    end
    return y2;
end

function batchImgTo3D(y, param)
    y2 = zeros(y);
    for i in 1:size(y,2)
        y2[:,i] = jointsImgTo3D(y[:,i], param);
    end
    return y2;
end

#= Denormalize sample from metric 3D to image coordinates
:sample: joints in (x,y,z) with x,y and z in mm
:param: camera intirinsic parameters
:return: joints in (x,y,z) with x,y in image coordinates and z in mm
=#
function joint3DToImg(sample, param)
    ret = Array{Float32}(3);
    ret[1] = sample[1]/sample[3]*param[1] + param[3];
    ret[2] = sample[2]/sample[3]*param[2] + param[4];
    ret[3] = sample[3];
    return ret
end

function joints3DToImg(y, param)
    if length(y)%3 != 0
        println("Error: Y should be a multiple of 3");
        return;
    end
    y2 = Array{Float32}(length(y));
    for i = 1:3:length(y)
        y2[i:i+2] = joint3DToImg(y[i:i+2], param);
    end
    return y2;
end

function batch3DToImg(y, param)
    y2 = zeros(y);
    for i in 1:size(y,2)
        y2[:,i] = joints3DToImg(y[:,i], param);
    end
    return y2;
end

# it converts the float pixel value to depth in mm
function floatToMm(x)
    if length(x) > 1
        xi = convert(Array{Int32, ndims(x)},(floor.(x.*65536)));
    else
        xi = convert(Int32,(floor(x*65536))); # TODO need floor??
    end
    return xi;
end

# it converts a mm lenght to float pixel value.
function mmToFloat(x)
    if length(x) > 1
        xi = convert(Array{Float32, ndims(x)}, x)./65536;
    else
        xi = convert(Float32, x)/65536;
    end
    return xi;
end

function transformPoint2D(pt, M)
    pt2 = map(Float32, M * [pt[1], pt[2], 1])
    return [pt2[1] / pt2[3], pt2[2] / pt2[3]]
end

# M applies scaling and translation to com
function transformJoints2D(y, M)
    if length(y)%3 != 0
        println("Error: Y should be a multiple of 3");
        return;
    end
    y2 = copy(y);
    for i = 1:3:length(y)
        y2[i:i+1] = transformPoint2D(y[i:i+1], M);
    end
    return y2;
end

# dset =0 ICVL, 1->NYU
function convertCrop3DToImg(com3D, joints3DCrop, dset)
    if dset == 0
        joints3D = copy(joints3DCrop.*250);
        param = (241.42, 241.42, 160., 120.);
    else
        joints3D = copy(joints3DCrop.*300);
        param = (588.03, -587.07, 320., 240.)
    end
    for i in 1:3:size(joints3DCrop,1)
        joints3D[i] += com3D[1];
        joints3D[i+1] += com3D[2];
        joints3D[i+2] += com3D[3];
    end
    return joints3DToImg(joints3D, param)
end
