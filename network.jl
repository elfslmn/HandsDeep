using Knet;
using MultivariateStats
#include("transformation.jl");

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

function initEmbed(inputdim, outputdim, Atype, embedSize, y)
    w = initBase(inputdim, outputdim, Atype);
    pop!(w); pop!(w); # remove last layer params

    # embeding layer
    push!(w, xavier(embedSize,1024))
    push!(w, zeros(embedSize,1))

    # last fully connected - reconstruction
    prior_w = getPcaComponents(embedSize, ytrn);
    push!(w, prior_w)
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

    #second conv layer (5,5)x8 with (2,2) pooling
    (x21,x22,cx2) =(inputdim[2], inputdim[2], 1) ;
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx2,cy))
    push!(w, zeros(1,1,cy,1))
    x2 = (div(x21-w1+1,2),div(x22-w2+1,2),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2

    #third conv layer (3,3)x8 with no pooling
    (x31,x32,cx3) = (inputdim[3], inputdim[3], 1) ;
    (w1,w2,cy) = (3,3,8);
    push!(w, xavier(w1,w2,cx3,cy))
    push!(w, zeros(1,1,cy,1))
    x3 = (div(x31-w1+1,1),div(x32-w2+1,1),cy)

    # 512 fully connected
    s = prod(x1) + prod(x2) + prod(x3);
    push!(w, xavier(512,s))
    push!(w, zeros(512,1))

    # output layer
    push!(w, xavier(outputdim,512))
    push!(w, zeros(outputdim,1))

    return map(Atype, w)
end

#Filter and pool sizes according to implementation(scalenet.py) not same as stated in paper
function initScale(inputdim, outputdim, Atype; resizeFactor = 2)
    w = Any[]
    # 1 - full scale (128,128,1)
    (x1,x2,cx) = inputdim;
    #first conv layer (5,5)x8 with (4,4) pooling
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    (x1,x2,cx) = (div(x1-w1+1,4),div(x2-w2+1,4),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=3

    #second conv layer (5,5)x8 with (2,2) pooling
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    (x1,x2,cx) = (div(x1-w1+1,2),div(x2-w2+1,2),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=3

    #third conv layer (3,3)x8 , no pooling
    (w1,w2,cy) = (3,3,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    x_full = (div(x1-w1+1,1),div(x2-w2+1,1),cy)
    info(x_full)

# 2 - half scale (64,64,1)
    (x1,x2,cx) = (inputdim[1]/resizeFactor, inputdim[2]/resizeFactor, inputdim[3]) ;
    #first conv layer (5,5)x8 with (2,2) pooling
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    (x1,x2,cx) = (div(x1-w1+1,2),div(x2-w2+1,2),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=3

    #second conv layer (5,5)x8 with (2,2) pooling
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    (x1,x2,cx) = (div(x1-w1+1,2),div(x2-w2+1,2),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=3

    #third conv layer (3,3)x8 , no pooling
    (w1,w2,cy) = (3,3,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    x_half = (div(x1-w1+1,1),div(x2-w2+1,1),cy)
    x_half = map(Int, x_half);
    info(x_half)

# 3 - quarter scale (32,32,1)
    (x1,x2,cx) = (inputdim[1]/(resizeFactor^2), inputdim[2]/(resizeFactor^2), inputdim[3]) ;
    #first conv layer (5,5)x8 with (2,2) pooling
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    (x1,x2,cx) = (div(x1-w1+1,2),div(x2-w2+1,2),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=3

    #second conv layer (5,5)x8 with no pooling
    (w1,w2,cy) = (5,5,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    (x1,x2,cx) = (div(x1-w1+1,1),div(x2-w2+1,1),cy)  # assuming conv4 with p=0, s=1 and pool with p=0,w=s=3

    #third conv layer (3,3)x8 , no pooling
    (w1,w2,cy) = (3,3,8);
    push!(w, xavier(w1,w2,cx,cy))
    push!(w, zeros(1,1,cy,1))
    x_quar = (div(x1-w1+1,1),div(x2-w2+1,1),cy)
    x_quar = map(Int, x_quar);
    info(x_quar)

# concanete all scales
    # 1024 fully connected 1
    s = prod(x_full) + prod(x_half) + prod(x_quar);
    push!(w, xavier(1024,s))
    push!(w, zeros(1024,1))

    # 1024 fully connected 2
    push!(w, xavier(1024,1024))
    push!(w, zeros(1024,1))

    # last fully connected
    push!(w, xavier(outputdim,1024))
    push!(w, zeros(outputdim,1))

    return map(Atype, w)
end


function baseNet(w,x; drop = true);
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

    #dropout
    x = drop ? dropout(x,0.3) : x ;

    # 1024 fully connected 2
    x = w[9]*mat(x) .+ w[10]
    x = relu.(x)

    #dropout
    x = drop ? dropout(x,0.3) : x ;

    # last fully connected
    x = w[11]*mat(x) .+ w[12]

    return x
end

function embedNet(w,x; drop = true)
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

    #dropout
    x = drop ? dropout(x,0.3) : x ;

    # 1024 fully connected 2
    x = w[9]*mat(x) .+ w[10]
    x = relu.(x)

    #dropout
    x = drop ? dropout(x,0.3) : x ;

    # embeding layer
    x = w[11]*mat(x) .+ w[12]
    x = relu.(x)

    # last fully connected
    x = w[13]*mat(x) .+ w[14]
    return x
end

function scaleNet(w,x_all; drop = true)
    # 1 - full scale (128,128,1)
    #first conv layer (5,5)x8 with (4,4) pooling
    x = conv4(w[1],x_all[1]) .+ w[2]
    x = pool(relu.(x); window =4)

    #second conv layer (5,5)x8 with (2,2) pooling
    x = conv4(w[3],x) .+ w[4]
    x = pool(relu.(x); window =2)

    #third conv layer (3,3)x8 , no pooling
    x = conv4(w[5],x) .+ w[6]
    x_full = relu.(x)
    info(size(mat(x_full)))

# 2 - half scale (64,64,1)
    #first conv layer (5,5)x8 with (2,2) pooling
    x = conv4(w[7],x_all[2]) .+ w[8]
    x = pool(relu.(x); window =2)

    #second conv layer (5,5)x8 with (2,2) pooling
    x = conv4(w[9],x) .+ w[10]
    x = pool(relu.(x); window =2)

    #third conv layer (3,3)x8 , no pooling
    x = conv4(w[11],x) .+ w[12]
    x_half = relu.(x)
    info(size(x_half))

# 3 - quarter scale (32,32,1)
    #first conv layer (5,5)x8 with (2,2) pooling
    x = conv4(w[13],x_all[3]) .+ w[14]
    x = pool(relu.(x); window =2)

    #second conv layer (5,5)x8 with no pooling
    x = conv4(w[15],x) .+ w[16]
    x = relu.(x)

    #third conv layer (3,3)x8 , no pooling
    x = conv4(w[17],x) .+ w[18]
    x_quar= relu.(x)
    info(size(x_quar))

# concanete all scales
    x = vcat(mat(x_full), mat(x_half), mat(x_quar));
    info(size(x))
    # 1024 fully connected 1
    x = w[19]*mat(x) .+ w[20]
    x = relu.(x)

    # 1024 fully connected 2
    x = w[21]*mat(x) .+ w[22]
    x = relu.(x)

    # last fully connected
    x = w[23]*mat(x) .+ w[24]
    return x;
end

#TODO complete
function refineNet(w,x; drop = true)
    x64 = x[1];
    x32 = x[2];
    x16 = x[3];
    #largest conv layer (5,5)x8 with (2,2) pooling
    x64 = conv4(w[1],x64) .+ w[2]
    x64 = pool(relu.(x64); window =2)

    #middle conv layer (5,5)x8 with (2,2) pooling
    x32 = conv4(w[3],x32) .+ w[4]
    x32 = pool(relu.(x32); window =2)

    #small conv layer (3,3)x8 with no pooling
    x16 = conv4(w[5],x16) .+ w[6]
    x16 = relu.(x16);

    # 1024 fully connected 1ayer
    #concanete all scales
    x = vcat(mat(x64), mat(x32), mat(x16));
    x = w[7]*mat(x) .+ w[8]
    x = relu.(x)

    # output layer
    x = w[9]*mat(x) .+ w[10]

    return x
end

function euc_loss(w, x, truth, model)
    pred = model(w,x);
    dist = 0;
    #=for j in 1:size(pred,2)
        for i in 1:3:size(pred,1)
            dist += sqrt((truth[i,j]-pred[i,j])^2 + (truth[i+1,j]-pred[i+1,j])^2 + (truth[i+2,j]-pred[i+2,j])^2);
        end
    end =#
    for j in 1:size(pred,2)
        for i in 1:size(pred,1)
            dist += (truth[i,j]-pred[i,j])^2
        end
    end

    return dist / size(pred,2);
end

#euc_loss_all(w,data, model) = mean(euc_loss(w,x,y, model) for (x,y) in data)

function euc_loss_all(w,data, model)
    s, n = 0.0, 0
    for (x,y) in data
        loss = euc_loss(w,x,y, model);
        if !isnan(loss)
            s += loss
            n += 1
        else
            println("Loss in nan")
        end
    end
    return s/n;
end


function huber(a, b; delta = 100)
    diff = abs(a-b);
    return delta^2 *(sqrt(1+(diff/delta)^2) -1);
end

function huber_loss(w, x, truth, model; param = (241.42, 241.42, 160., 120.))
    pred = model(w,x,; drop = false);
    r = mean(huber.(pred, truth))
    return r
end

function l2reg(w,lambda)
    J = Float32(0);
    J += Float32(lambda) * sum(sum(abs2,wi) for wi in w[1:2:end]);
    return J
end

function objective(w, x, truth, model, l2)
    h = euc_loss(w, x, truth, model);
    l2 = l2reg(w, l2)
    return (h+l2)
end

lossgradient= grad(objective);

function train_sgd(w, dtrn, net, opt; l2=0.001)
    if net == refineNet
        for (x64,y) in dtrn[1], (x32,y) in dtrn[2], (x16,y) in dtrn[3]
            gr = lossgradient(w,(x64,x32,x16) ,y, net, l2)
            update!(w, gr, opt)
        end
    else
        for (x,y) in dtrn
            gr = lossgradient(w,x,y, net, l2)
            update!(w, gr, opt);
        end
    end
end

function accuracy_batch(w, x,y, threshold, net)
    positive =0;
    pred = net(w,x; drop = false);
    for j in 1:size(pred,2)
        for i in 1:3:size(pred,1)
            dist = sqrt((y[i,j]-pred[i,j])^2 + (y[i+1,j]-pred[i+1,j])^2 + (y[i+2,j]-pred[i+2,j])^2);
            if dist>threshold
                break;
            end
            if i == (size(pred,1) -2)
                positive +=1;
            end
        end
    end
    return positive / size(y,2)
end

function accuracy_all(w, data, threshold, net)
    all = 0;
    positive =0;
    for (x,y) in data
        all += size(y,2) ;
        positive += accuracy_batch(w, x,y, threshold, net)*size(y,2) ;
    end
    return positive / all
end

#accuracy_all(w, data, threshold, net) = mean(accuracy_batch(w, x,y, threshold, net) for (x,y) in data)

function getPcaComponents(sz, y)
    M = fit(PCA, map(Float32,y); pratio=1.);
    return projection(M)[:,1:sz]
end

#return avarage joint distance error(in mm) and accuracy
# y should be in 3d world coordinates and normalized(./250)
function getLossandAccuracy(w,data,threshold, net; dset = 0)
    dist = 0;
    positive =0;
    all = data.length;
    for (x,y) in data
        pred = net(w,x; drop = false);

        scale = dset == 0 ? 250:300;
        pred3D = pred.*scale;
        y3D = y.*scale;

        for j in 1:size(pred,2)
            pass = true;
            for i in 1:3:size(pred,1)
                d = sqrt((y3D[i,j]-pred3D[i,j])^2 + (y3D[i+1,j]-pred3D[i+1,j])^2 + (y3D[i+2,j]-pred3D[i+2,j])^2);
                dist +=d/(size(pred,1)/3); # to calculate mean distance of all joints
                if d>threshold
                    pass = false;
                end
            end
            if pass
                positive +=1;
            end
        end

    end
    return (dist/all , positive/all*100 )
end

function getRefineLoss(w,data)
    dist = 0;
    all = data[1].length; print("all=",all);
    for (x64,y) in data[1], (x32,y) in data[2], (x16,y) in data[3]
        pred = refineNet(w,(x64,x32,x16); drop = false);

        for j in 1:size(pred,2)
            d = sqrt((y[1,j]-pred[1,j])^2 + (y[2,j]-pred[2,j])^2 + (y[3,j]-pred[3,j])^2);
            dist += d; # to calculate mean distance of all joints
        end
    end
    return (dist/all)
end

function getMeanErrorOfEachJoint(y, pred; dset= 0)
    nSample = size(y,2);
    nJoint = convert(Int64, size(y,1)/3);

    scale = dset == 0 ? 250:300;
    pred3D = pred.*scale;
    y3D = y.*scale;

    err = zeros(nJoint)
    for j in 1:nSample
        for i in 1:nJoint
            err[i] += sqrt((y3D[3(i-1)+1,j]-pred3D[3(i-1)+1,j])^2 +
                            (y3D[3(i-1)+2,j]-pred3D[3(i-1)+2,j])^2 +
                            (y3D[3(i-1)+3,j]-pred3D[3(i-1)+3,j])^2);
        end
    end
    err = err./nSample;
    return err;
end
