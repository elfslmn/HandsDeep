using ImageView, Images;
#include("transformation.jl");

# joints and pred should be unnormalized
function showAnnotation(img,joints; pred = Any[])
    if typeof(img[1,1]) != Gray{N0f16}
        if typeof(img[1,1]) == Int16 #NYU images
            img = normalizeDepth(map(Float32, img); a=0,b=1);
        end
        imgg = convert(Array{Gray{N0f16},ndims(img)}, img);
    end
    m = convert(Float32, maximum(imgg));
    img3 = RGB.(imgg);
    w = size(img,2);
    h = size(img,1);
    for i in 1:3:length(joints)
        xr = round(Int,joints[i]);
        xl = xr + 5;
        yt = round(Int,joints[i+1]);
        # print("x,y="); print(xr); print("-"); println(yt);
        yb = yt + 5;
        if xl > w
            xl = w;
        end
        if yb>h
            yb = h;
        end
        img3[yt:yb, xr:xl] = RGB{N0f16}(m ,0.,0.);
    end
    # show prediction
    if size(pred,1) != 0
        for i in 1:3:length(pred)
            xr = round(Int,pred[i]);
            xl = xr + 5;
            yt = round(Int,pred[i+1]);
            # print("x,y="); print(xr); print("-"); println(yt);
            yb = yt + 5;
            if xl > w
                xl = w;
            end
            if yb>h
                yb = h;
            end
            img3[yt:yb, xr:xl] = RGB{N0f16}(0.,m,0.);
        end
    end
    imshow(img3);
end

#= Calculate the center of mass
:param dpt: depth image (ICVL ->float32, NYU -> int16 )
:return: (x,y,z) center of mass x,y in pixels, z is in mm
:dset = 0 -> ICVL,  type=1 -> NYU =#

function calculateCoM(dpt; dset = 0)
    dc = copy(dpt);
    if dset == 0 #ICVL
        minz = mmToFloat(10); # hand can be in the range between 10mm to 1500mm
        maxz = mmToFloat(1000);
        dc[dc .< minz] = 0.;
        dc[dc .> maxz] = 0.;
    else #NYU
        minz = 0;
        maxz = 1000;
        dc[dc .< minz] = 0;
        dc[dc .> maxz] = 0;
    end

    num = countnz(dc);
    if num == 0
        return NaN;
    end
    x=0.; y=0.; z=0.;

    for c in 1:size(dc,2) # TODO ???????*
        for r in 1:size(dc,1)
            w = dc[r,c];
            if(w > 0)
                x += c
                y += r
                z += w
            end
        end
    end
    if dset == 0
        return [x/num, y/num, floatToMm(z/num)]
    else
        return [x/num, y/num, z/num]
    end
end


#=  Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
:com: center of mass, in image coordinates (x,y,z), z in mm
:size: (x,y,z) extent of the source crop volume in mm
:param: camera intirinsics
:return: xstart, xend, ystart, yend, zstart, zend (x,y in pixels, z in mm)=#
function comToBounds(com, size, param)
    zstart = convert(Int64, floor(com[3] - size[3] / 2.));
    zend = convert(Int64, floor(com[3] + size[3] / 2.));
    xstart = convert(Int64, floor((com[1] * com[3] / param[1] - size[1] / 2.) / com[3]*param[1]))
    xend = convert(Int64, floor((com[1] * com[3] / param[1] + size[1] / 2.) / com[3]*param[1]))
    ystart = convert(Int64, floor((com[2] * com[3] / param[2] - size[2] / 2.) / com[3]*param[2]))
    yend = convert(Int64, floor((com[2] * com[3] / param[2] + size[2] / 2.) / com[3]*param[2]))
    r = [xstart, ystart, zstart] , [xend, yend, zend];
    return r
end

function getCrop(dpt, xstart, xend, ystart, yend, zstart, zend; cropz = true, dset= 0)
    cropped = dpt[max(ystart, 1):min(yend, size(dpt,1)), max(xstart, 1):min(xend, size(dpt,2))]

    if cropz
        if dset == 0
            zstartf = mmToFloat(zstart);
            zendf = mmToFloat(zend);
        else
            zstartf = zstart;
            zendf = zend;
        end
        cropped[cropped .< zstartf] = zstartf;
        cropped[cropped .> zendf] = zendf;
    else
        zendf = 0; # pad with 0
    end

    # to keep the w/h ratio
    padded = parent(padarray(cropped, Fill(zendf,(abs(ystart)-max(ystart, 0),abs(xstart)-max(xstart, 0))
                    ,(abs(yend)-min(yend, size(dpt,1)), abs(xend)-min(xend, size(dpt,2))))))
    return padded
end

#:dset = 0 -> ICVL,  type=1 -> NYU =#
function extractHand(dpt, param, imgSize; dset = 0)
    com = calculateCoM(dpt; dset = dset);
    if any(isnan, com)
        return NaN, NaN;
    end
    if dset == 0 #ICVL
        st, fn = comToBounds(com, (250,250,250), param)
    else #NYU
        st, fn = comToBounds(com, (300,300,300), param)
    end
    #showAnnotation(dpt, vcat(com,st,fn))
    p = getCrop(dpt, st[1], fn[1], st[2], fn[2], st[3], fn[3]; dset=dset);
    return imresize(p,(imgSize,imgSize)), com ;
end

# normalize between -1 and 1;
function normalizeDepth(dpt; a=-1, b=1)
    dptc = copy(dpt);
    maxz = maximum(dpt);
    minz = minimum(dpt);
    #dptc = (dptc .- minz)./(0.5*(maxz-minz)) .- 1;
    dptc = (b-a) * (dptc .- minz)./(maxz-minz) .+a;
    return dptc;
end

#:dset = 0 -> ICVL,  type=1 -> NYU =#
# img should be Array{Float32,2} -> ICVL
function preprocess(img, param, imgSize; dset = 0)
    (hd , com) = extractHand(img, param, imgSize; dset = dset);
    if dset == 0
        centerz = mmToFloat(com[3])
        cube = mmToFloat(250/2);
    else
        centerz = com[3]
        cube = 300/2;
    end
    # normalize
    hd = (hd.- centerz)./ cube;
    return hd, com;
end

#:dset = 0 -> ICVL,  type=1 -> NYU =#
function extractPatch(img, center, dim; dset = 0)
    xstart = convert(Int, floor(center[1] - dim[1]/2))+1;
    xend = xstart + dim[1]-1;
    ystart = convert(Int, floor(center[2] - dim[2]/2))+1;
    yend = ystart + dim[2]-1;
    p = getCrop(img,xstart, xend, ystart, yend, 0, 0; cropz = false, dset = dset);
    return p
end
