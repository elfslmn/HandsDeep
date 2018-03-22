# Weight initialization for multiple layers
# h[i] is an integer for a fully connected layer, a triple of integers for convolution filters
# Output is an array [w0,b0,w1,b1,...,wn,bn] where wi,bi is the weight matrix/tensor and bias vector for the i'th layer
function cinit(h...)  # use cinit(x,h1,h2,...,hn,y) for n hidden layer model
    w = Any[]
    x = h[1]
    for i=2:length(h)
        if isa(h[i],Tuple) #conv layer
            (x1,x2,cx) = x
            (w1,w2,cy) = h[i]
            push!(w, xavier(w1,w2,cx,cy))
            push!(w, zeros(1,1,cy,1))
            x = (div(x1-w1+1,2),div(x2-w2+1,2),cy) # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2
        elseif isa(h[i],Integer) # fully connected layer
            push!(w, xavier(h[i],prod(x)))
            push!(w, zeros(h[i],1))
            x = h[i]
        else
            error("Unknown layer type: $(h[i])")
        end
    end
    map(Atype, w)
end

function convnet(w,x; pdrop=(0,0,0))    # pdrop[1]:input, pdrop[2]:conv, pdrop[3]:fc
    for i=1:2:length(w)
        if ndims(w[i]) == 4     # convolutional layer
            x = dropout(x, pdrop[i==1?1:2])
            x = conv4(w[i],x) .+ w[i+1]
            x = pool(relu.(x))
        elseif ndims(w[i]) == 2 # fully connected layer
            x = dropout(x, pdrop[i==1?1:3])
            x = w[i]*mat(x) .+ w[i+1]
            if i < length(w)-1; x = relu.(x); end
        else
            error("Unknown layer type: $(size(w[i]))")
        end
    end
    return x
end

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

function euc_loss(w, x, truth)
    pred = baseNet(w,x);
    return sum((truth .- pred).^2) /(size(pred,2))
end
euc_loss_all(w,data) = mean(euc_loss(w,x,y) for (x,y) in data)

lossgradient= grad(euc_loss);

function train_sgd(w, dtrn, lr) #lr= learning rate, dtrn= all training data
    # YOUR CODE HERE
   for (x,y) in dtrn
        gr = lossgradient(w,x,y)
        for i in 1:length(w)
            w[i] -= lr * gr[i]
        end
    end

end
