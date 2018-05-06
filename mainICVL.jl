using Knet,JLD;
include("network.jl")
include("transformation.jl");
include("util.jl");
include("ICVLreader.jl")

if isfile(joinpath(pwd(),"icvl_plus.jld"))
    dict = load("icvl_plus.jld"); #TODO
    xtrn = dict["xtrn"];
    ytrn = dict["ytrn"];
    comstrn = dict["comstrn"];
    trMatstrn = dict["trMatstrn"];
    xtst = dict["xtst"];
    ytst = dict["ytst"];
    comstst = dict["comstst"];
    trMatstst = dict["trMatstst"];
    clear!(:dict)
else
    xtrn, ytrn, comstrn, trMatstrn, imgst = readICVLTraining(;raw=true, sz=128 );
    xtst, ytst, comstst, trMatstst, imgs = readICVLTesting(;raw=true);
    save(joinpath(pwd(),"icvl_plus.jld"),
     "xtrn", xtrn,"ytrn",ytrn,"comstrn",comstrn,"trMatstrn",trMatstrn,
    "xtst",xtst,"ytst",ytst, "comstst",comstst, "trMatstst", trMatstst);
end

EPOCHS = 100;
LR = 0.01;
BATCHSIZE = 128;
THRESHOLD = 60;
EMBEDING = 30;
INPUTDIM = (128,128,1);
OUTPUTDIM = 48;
PARAM = (241.42, 241.42, 160., 120.) # ICVL
Atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}


# Minibatch data
dtst = minibatch(xtst,ytst,BATCHSIZE;xtype=Atype, ytype=Atype)
dtrn = minibatch(xtrn,ytrn,BATCHSIZE;xtype=Atype, ytype=Atype)
length(dtrn),length(dtst)
clear!(:xtrn); clear!(:xtst);


function report(epoch, w_emb, net, pr, m)
    w = deepcopy(w_emb);
    push!(w, pr);
    push!(w, m);
    (l1 , a1) = getLossandAccuracy(w,dtrn,THRESHOLD, net)
    (l2 , a2) = getLossandAccuracy(w,dtst,THRESHOLD, net)
    push!(trn_loss, l1);    push!(trn_acc, a1);
    push!(tst_loss, l2);    push!(tst_acc, a2);
    println((:epoch,epoch,:Trnlss,l1,:a,a1,:Tstlss,l2,:a,a2 ))
end

#Embeding training
# train , input -> embeding --------------------------------------------------
M = fit(PCA, map(Float32,ytrn); pratio=1.);
pr = projection(M)[:,1:EMBEDING]
m = mean(M);
ytrn_emb = (pr')*(ytrn.-m);

dtrn_emb = minibatch(xtrn,ytrn_emb,BATCHSIZE;xtype=Atype, ytype=Atype)
dtrn = minibatch(xtrn,ytrn,BATCHSIZE;xtype=Atype, ytype=Atype)
dtst = minibatch(xtst,ytst,BATCHSIZE;xtype=Atype, ytype=Atype)

trn_loss = Any[]; trn_acc = Any[];
tst_loss = Any[]; tst_acc = Any[]

w_emb = initBase(INPUTDIM, EMBEDING, Atype);
opt = optimizers(w_emb, Rmsprop; lr=LR, gclip=0.01)

report(0,w_emb,embedNet,pr,m);
@time for epoch=1:EPOCHS
    lrate = LR/(1+0.2*(epoch-1));
    for i in 1:size(opt,1)
        opt[i].lr = lrate;
    end
    train_sgd(w_emb, dtrn_emb, baseNet, opt)
    report(epoch,w_emb,embedNet,pr,m);
end


acctst = Any[]
function reportTesting(epoch, w,net, thresh)
    (euc_tst , a) = getLossandAccuracy(w,dtst,thresh, net)
    push!(acctst, a)
    println((:thresh,thresh,:TstLoss,euc_tst,:accuracy, a))
end


for t in 10:10:80
    reportTesting(0, w_emb, embedNet, t);
end


using Plots
plot(1:100,hcat(dict["trn_acc"], dict["tst_acc"]),label=["Training" "Test"],xlabel="epochs",ylabel="Fraction of frames within 60 mm distance %", lw=2, ylim=[0,100])
plot(1:100,hcat(dict["trn_loss"],dict["tst_loss"]),label=["Training" "Test"],xlabel="epochs",ylabel="Loss", lw=2,ylim=[30,55])
plot(10:10:80,hcat(dict["th_trn"],dict["th_tst"]),label=["Training" "Test"],xlabel="Distance threshold /mm",ylabel="Fraction of frames within distance %", lw=2, ylim=[0,100])







#refineNet training -------------------------------------------------------------

x = getICVLJointImages();
srand(1);
perm = randperm(size(x,4));
x64 = x[:,:,:,perm,:];
y = ytrn[:,perm];

EPOCHS = 50;
LR = 0.01;
BATCHSIZE = 128;
EPOCHREG = 0.2
L2REG = 0.0001
Atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
INPUTDIM = (64,32,16);
OUTPUTDIM = 3;
JOINT = 16;

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


pred = zeros(ytst)
i=0;
for (x,y) in dtst
    if i == length(dtst)-1
        pred[:,128*i+1:end] = convert(Array{Float32},embedNet(w,x; drop = false))
    else
        pred[:,128*i+1:128*(i+1)] = convert(Array{Float32},embedNet(w,x; drop = false))
    end
    i += 1;
end

err = getMeanErrorOfEachJoint(ytst, pred; dset=0);

err_ref= Any[]
for j=1:16
    jin = 3*(j-1)+1:3*j;
    (_,p,r)= refineJointIterativeBatch(imgs, pred[jin,:], comstst,ytst[jin,:], w_refs[j], 1, 0);
    push!(err_ref, r)
    println(j,"\t",p,"\t",r);
end
