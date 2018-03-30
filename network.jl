function initBase(inputdim, outputdim, Atype)
    w = Any[]
    (x1,x2,cx) = inputdim;
    #first conv layer (5,5)x8 with (3,3) pooling
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    (x1,x2,cx) = (div(x1-w1+1,3),div(x2-w2+1,3),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=3

    #second conv layer (5,5)x8 with (3,3) pooling
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    (x1,x2,cx) = (div(x1-w1+1,3),div(x2-w2+1,3),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=3

    #third conv layer (3,3)x8 , no pooling
    (w1,w2,cy) = (3,3,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    x = (div(x1-w1+1,1),div(x2-w2+1,1),cy)

    # 1024 fully connected 1
    push!(w, xavier(1024,prod(x)))
    push!(w, zeros(1024,1))

    # 1024 fully connected 2
    push!(w, xavier(1024,1024))
    push!(w, zeros(1024,1))

    # last fully connected
    push!(w, xavier(outputdim,1024))
    push!(w, zeros(outputdim,1))

    return map(Atype, w)
end

function initEmbed(inputdim, outputdim, Atype, embedSize)
    w = initBase(inputdim, outputdim, Atype);
    pop!(w); pop!(w); # remove last layer params

    # embeding layer
    push!(w, xavier(embedSize,1024))
    push!(w, zeros(embedSize,1))

    # last fully connected
    push!(w, xavier(outputdim,embedSize))
    push!(w, zeros(outputdim,1))
    return map(Atype, w)
end

function initRefine(inputdim, outputdim, Atype)
    w = Any[]
    #first conv layer (5,5)x8 with (2,2) pooling
    (x11,x12,cx1) = (inputdim[1], inputdim[1], 1) ;
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx1,cy))
    push!(w, zeros(1,1,cy,1))
    x1 = (div(x11-w1+1,2),div(x12-w2+1,2),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2
    info(x1);

    #second conv layer (5,5)x8 with (2,2) pooling
    (x21,x22,cx2) =(inputdim[2], inputdim[2], 1) ;
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx2,cy))
    push!(w, zeros(1,1,cy,1))
    x2 = (div(x21-w1+1,2),div(x22-w2+1,2),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2
    info(x2);

    #third conv layer (3,3)x8 with no pooling
    (x31,x32,cx3) = (inputdim[3], inputdim[3], 1) ;
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx3,cy))
    push!(w, zeros(1,1,cy,1))
    x3 = (div(x31-w1+1,1),div(x32-w2+1,1),cy)
    info(x3);

    # 1024 fully connected 1
    s = prod(x1) + prod(x2) + prod(x3);
    push!(w, xavier(1024,s))
    push!(w, zeros(1024,1))

    # last fully connected
    push!(w, xavier(outputdim,1024))
    push!(w, zeros(outputdim,1))

    return map(Atype, w)
end

function baseNet(w,x)
    #first conv layer (5,5)x8 with (3,3) pooling
    x = conv4(w[1],x) .+ w[2]
    x = pool(relu.(x); window =3)

    #second conv layer (5,5)x8 with (3,3) pooling
    x = conv4(w[3],x) .+ w[4]
    x = pool(relu.(x); window =3)

    #third conv layer (3,3)x8 , no pooling
    x = conv4(w[5],x) .+ w[6]
    x = relu.(x)

    # 1024 fully connected 1
    x = w[7]*mat(x) .+ w[8]
    x = relu.(x)

    # 1024 fully connected 2
    x = w[9]*mat(x) .+ w[10]
    x = relu.(x)

    # last fully connected
    x = w[11]*mat(x) .+ w[12]

    return x
end

function embedNet(w,x)
    #first conv layer (5,5)x8 with (3,3) pooling
    x = conv4(w[1],x) .+ w[2]
    x = pool(relu.(x); window =3)

    #second conv layer (5,5)x8 with (3,3) pooling
    x = conv4(w[3],x) .+ w[4]
    x = pool(relu.(x); window =3)

    #third conv layer (3,3)x8 , no pooling
    x = conv4(w[5],x) .+ w[6]
    x = relu.(x)

    # 1024 fully connected 1
    x = w[7]*mat(x) .+ w[8]
    x = relu.(x)

    # 1024 fully connected 2
    x = w[9]*mat(x) .+ w[10]
    x = relu.(x)

    # embeding layer
    x = w[11]*mat(x) .+ w[12]
    x = relu.(x)

    # last fully connected
    x = w[13]*mat(x) .+ w[14]
    return x
end

function extractPatch(img, center, dim)
    patch = zeros(Float32, dim,dim,1,size(img,4));
    (px, py) = center;
    half = Int64.(dim /2 -0.5);
    for i in px-half : px+half
        if i>size(img, 2) || i <= 0
            continue;
        end
        for j in py-half : py+half
            if j>size(img, 1) || j <= 0
                continue;
            end

        end
    end
end

function refineNet(w,x, patchSizes, center)
    (px, py) = center;
    s1 = Int64.(patchSizes[1] /2 -0.5);
    x1 = x[px-s1:px+s1, py-s1:py+s1, :, :];
    println(summary(x1));

    #first conv layer (5,5)x8 with (2,2) pooling

    #second conv layer (5,5)x8 with (2,2) pooling

    #third conv layer (3,3)x8 with no pooling

    # 1024 fully connected 1ayer

    # last fully connected

    return x
end



function euc_loss(w, x, truth, model)
    pred = model(w,x);
    return ( sum((truth .- pred).^2) / (size(pred,2)) )
end
euc_loss_all(w,data, model) = mean(euc_loss(w,x,y, model) for (x,y) in data)

lossgradient= grad(euc_loss);

function train_sgd(w, dtrn, lr, net) #lr= learning rate, dtrn= all training data
    # YOUR CODE HERE
   for (x,y) in dtrn
        gr = lossgradient(w,x,y, net)
        for i in 1:length(w)
            w[i] -= lr * gr[i]
        end
    end

end


function accuracy_batch(w, x,y, threshold, net)
    positive =0;
    pred = net(w,x);
    for i in 1:3:size(pred,1)
        dist = sqrt((y[i]-pred[i])^2 + (y[i+1]-pred[i+1])^2 + (y[i+2]-pred[+2])^2);
        if dist>threshold
            break;
        end
        if i == (size(pred,1) -2)
            positive +=1;
        end
    end
    return positive / size(y,2)
end

#= function accuracy_all(w, data, threshold, net)
    all = 0;
    positive =0;
    for (x,y) in data
        all += size(y,2) ;
        positive += accuracy_batch(w, x,y, threshold, net)*size(y,2) ;
    end
    return positive / all
end =#

accuracy_all(w, data, threshold, net) = mean(accuracy_batch(w, x,y, threshold, net) for (x,y) in data)