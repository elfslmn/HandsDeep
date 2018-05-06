using Knet,JLD;
include("network.jl")
include("transformation.jl");
include("NYUreader.jl")
include("util.jl")

if isfile(joinpath(pwd(),"nyu_tst.jld"))
    dict = load("nyu_trn1.jld");
    xtrn = dict["xtrn"];
    ytrn = dict["ytrn"];
    dict = load("nyu_trn2.jld");
    xtrn = cat(4, xtrn, dict["xtrn"]);
    ytrn = cat(2, ytrn, dict["ytrn"]);

    dict = load("nyu_tst.jld");
    comstst = dict["comstst"];
    xtst = dict["xtst"];
    ytst = dict["ytst"];
    clear!(:dict)
else
    xtrn, ytrn, comstrn, trMatstrn = readNYUTraining();
    xtst, ytst, comstst, trMatstst = readNYUTesting();
    save(joinpath(pwd(),"nyu_trn2.jld"),
    "xtrn", xtrn,"ytrn",ytrn,"comstrn",comstrn,"trMatstrn",trMatstrn);
    save(joinpath(pwd(),"nyu_tst.jld"),
    "xtst",xtst,"ytst",ytst, "comstst",comstst, "trMatstst", trMatstst);

end

EPOCHS = 100;
LR = 0.01;
BATCHSIZE = 128;
THRESHOLD = 60;
EMBEDING = 30;
INPUTDIM = (128,128,1);
OUTPUTDIM = 42;
PARAM = (588.03, -587.07, 320., 240.)
Atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

function report(epoch, w_emb, net, pr, m)
    print("epoch ",epoch,":");
    w = deepcopy(w_emb);
    push!(w, pr);
    push!(w, m);
    (l1 , a1) = getLossandAccuracy(w,dtrn,THRESHOLD, net; dset=1)
    (l2 , a2) = getLossandAccuracy(w,dtst,THRESHOLD, net; dset=1)
    push!(trn_loss, l1);    push!(trn_acc, a1);
    push!(tst_loss, l2);    push!(tst_acc, a2);

    println((:Trnlss,l1,:a,a1,:Tstlss,l2,:a,a2 ))
end

# ========== TRAINING ITERATIONS ============
# train , input -> embeding --------------------------------------------------
M = fit(PCA, map(Float32,ytst); pratio=1.);
pr = projection(M)[:,1:EMBEDING]
m = mean(M);
ytrn_emb = (pr')*(ytst.-m);

# Minibatch data
srand(1);
perm = randperm(size(xtst,4));
dtrn_emb = minibatch(xtst[:,:,:,perm[513:end]],ytrn_emb[:,perm[513:end]],BATCHSIZE;xtype=Atype, ytype=Atype, shuffle=true) #use to train
dtrn = minibatch(xtst[:,:,:,perm[513:end]],ytst[:,perm[513:end]],BATCHSIZE;xtype=Atype, ytype=Atype, shuffle=true) #use to measure training acc
dtst = minibatch(xtst[:,:,:,perm[1:512]],ytst[:,perm[1:512]],BATCHSIZE;xtype=Atype, ytype=Atype)
length(dtrn),length(dtst)

trn_loss = Any[]
tst_loss = Any[]
trn_acc = Any[]
tst_acc = Any[]
w_emb = initBase(INPUTDIM, EMBEDING, Atype);
opt = optimizers(w_emb, Rmsprop; lr=LR, gclip=0.01)
report(0,w_emb,embedNet,pr,m);

@time for epoch=2:100
    lrate = LR/(1+0.2*(epoch-1));
    for i in 1:size(opt,1)
        opt[i].lr = lrate;
    end
    train_sgd(w_emb, dtrn_emb, baseNet, opt)
    report(epoch,w_emb,embedNet,pr,m);
end


function reportAcc(w,net, thresh, data, arr)
    (euc_tst , a) = getLossandAccuracy(w,data,thresh, net; dset=1)
    push!(arr, a)
    println((:thresh,thresh,:accuracy, a))
end


function distanceThreshold()
    w = deepcopy(w_emb);
    push!(w, pr);
    push!(w, m);

    th_trn = Any[];
    th_tst = Any[];
    for t in 10:10:80
        reportAcc(w, embedNet, t, dtrn, th_trn);
    end
    for t in 10:10:80
        reportAcc(w, embedNet, t, dtst, th_tst);
    end
end

function saveModel(filename)
    save(joinpath(pwd(),"nyu_exp1.jld"), "w",w, "trn_loss",trn_loss,"trn_acc",trn_acc ,
    "tst_loss", tst_loss, "tst_acc", tst_acc, "th_tst", th_tst);
end


function imsh(i)
    im = xtst[:,:,:,i]
    imshow(im);
    println(std(im))
end


#refineNet training -------------------------------------------------------------

x = getNYUJointImages();
srand(1);
perm = randperm(size(x,4));
x64 = x[:,:,:,perm,:];
y = ytrn[:,perm];

EPOCHS = 50;
LR = 0.01;
BATCHSIZE = 128;
EPOCHREG = 0.2
L2REG = 0.0001

INPUTDIM = (64,32,16);
OUTPUTDIM = 3;
JOINT = 14;

loss_all = Any[];
w_refs = Any[];
for d in 1:JOINT
    w = initRefine(INPUTDIM, OUTPUTDIM,Atype);
    opt = optimizers(w, Rmsprop; lr=LR, gclip=0.01)
    dtrn = minibatch(x64[:,:,:,:,d],y[3*(d-1)+1:3d,:],BATCHSIZE;xtype=Atype, ytype=Atype);
    loss=Any[];
    println("Joint ", d ," refine model is training..")
    ls = getRefineLoss(w,dtrn)
    println((:epoch , 0, :loss , ls))
    push!(loss, ls);
    @time for epoch=1:50
        lrate = LR/(1+EPOCHREG*(epoch-1));
        for i in 1:size(opt,1)
            opt[i].lr = lrate;
        end
        train_sgd(w, dtrn, refineNet, opt; l2=L2REG)
        ls = getRefineLoss(w,dtrn)
        push!(loss, ls);
        println((:epoch , epoch, :loss , ls))
    end
    push!(w_refs, map(Array{Float32}, w));
    push!(loss_all,loss);
end
save(joinpath(pwd(),"refs.jld"), "w_refs",w_refs, "loss_all", loss_all );
