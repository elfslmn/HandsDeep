using ImageView;

function showAnnotation(img,joints)
    if typeof(img[1,1]) == Float32
        imgg = convert(Array{Gray{N0f16},3}, img);
    end
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
        img3[yt:yb, xr:xl] = RGB{N0f16}(1.,0.,0.);
    end
    imshow(img3);
end
