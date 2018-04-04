#using ImageView;
#include("transformation.jl");

function showAnnotation(img,joints)
    if typeof(img[1,1]) == Float32
        imgg = convert(Array{Gray{N0f16},3}, img);
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
    imshow(img3);
end

#= Calculate the center of mass
:param dpt: depth image (float32)
:return: (x,y,z) center of mass x,y in pixels, z is in mm =#
function calculateCoM(dpt)
    min = mmToFloat(10); # hand can be in the range between 10mm to 1500mm
    max = mmToFloat(1500);

    dc = copy(dpt);
    dc[dc .< min] = 0.;
    dc[dc .> max] = 0.;

    num = countnz(dc);
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
    return [x/num, y/num, floatToMm(z/num)]
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

function getCrop(dpt, xstart, xend, ystart, yend, zstart, zend)
    cropped = dpt[max(ystart, 1):min(yend, size(dpt,1)), max(xstart, 1):min(xend, size(dpt,2))]
    zstartf = mmToFloat(zstart);
    zendf = mmToFloat(zend);
    cropped[cropped .< zstartf] = zstartf;
    cropped[cropped .> zendf] = zendf;

    # to keep the w/h ratio
    padded = parent(padarray(cropped, Fill(zendf,(abs(ystart)-max(ystart, 0),abs(xstart)-max(xstart, 0))
                    ,(abs(yend)-min(yend, size(dpt,1)), abs(xend)-min(xend, size(dpt,2))))))
    return padded
end

function extractHand(dpt, param)
    com = calculateCoM(dpt);
    st, fn = comToBounds(com, (250,250,250), param)
    #showAnnotation(dpt, vcat(com,st,fn))
    p = getCrop(dpt, st[1], fn[1], st[2], fn[2], st[3], fn[3]);
    return imresize(p,(128,128));
end

# normalize between -1 and 1;
function normalizeDepth(dpt)
    dptc = copy(dpt);
    maxz = maximum(dpt);
    minz = minimum(dpt);
    dptc = (dptc .- minz)./(0.5*(maxz-minz)) .- 1;
    return dptc;
end
# img should be Array{Float32,2}
function preprocess(img, param)
    hd = extractHand(img, param);
    return normalizeDepth(hd);
end
