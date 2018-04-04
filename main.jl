using Knet, ImageView, JLD;
include("network.jl")

if isfile(joinpath(pwd(),"icvl.jld"))
    dict = load("icvl.jld");
    xtrn = dict["xtrn"];
    ytrn = dict["ytrn"];
    xtst = dict["xtst"];
    ytst = dict["ytst"];
    clear!(:dict)
else
    include("ICVLreader.jl")
    xtst, ytst = readICVLTesting();
    xtrn, ytrn = readICVLTraining();
    save(joinpath(pwd(),"icvl.jld"), "xtrn", xtrn,"ytrn",ytrn,"xtst",xtst,"ytst",ytst);
end

EPOCHS = 100;
LR = 0.01;
BATCHSIZE = 128;
THRESHOLD = 1000; # not specified in the paper
EMBEDING = 8;
INPUTDIM = (240,320,1);
OUTPUTDIM = 48;
Atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

# Minibatch data
dtst = minibatch(xtst,ytst,BATCHSIZE;xtype=Atype)
dtrn = minibatch(xtrn,ytrn,BATCHSIZE;xtype=Atype)
length(dtrn),length(dtst)
clear!(:xtrn); clear!(:ytrn); clear!(:xtst); clear!(:ytst);


trn_loss = Any[]
tst_loss = Any[]

function report(epoch, w, data, net)
    euc_trn = euc_loss_all(w,data, net);
    push!(trn_loss, euc_trn)
    euc_tst = euc_loss_all(w,data, net)
    push!(tst_loss, euc_tst)
    a=accuracy_joint(w,data, THRESHOLD, net);
    println((:epoch,epoch,:trnloss,euc_trn,:tstloss,euc_tst, :acc, a))
end

#report(0,w, dtrn)
# ========== TRAINING ITERATIONS ============
#@time for epoch=1:EPOCHS #
#    train_sgd(w, dtrn, LR, baseNet)
#    report(epoch,w,dtrn)
#end


w_base = initBase(INPUTDIM, OUTPUTDIM, Atype);
println("Baseline Model");
for (x,y) in dtrn
    output= baseNet(w_base,x);
    println("Output array: " * summary(output));
    print("Loss: "); println(euc_loss(w_base,x,y, baseNet));
    print("Accuracy: ");
    println(accuracy_batch(w_base,x,y,THRESHOLD, baseNet));
    break; # run only for 1 minibatch for now
end

w_emb = initEmbed(INPUTDIM, OUTPUTDIM, Atype, EMBEDING);
println("Low Embeding Model");
for (x,y) in dtrn
    output= embedNet(w_emb,x);
    println("Output array: " * summary(output));
    print("Loss: "); println(euc_loss(w_emb,x,y, embedNet));
    print("Accuracy: ");
    println(accuracy_batch(w_emb,x,y,THRESHOLD, embedNet));
    break; # run only for 1 minibatch for now
end

patchSizes = [31,15,7]; #TODO not stated in the paper ??
w_ref = initRefine(patchSizes, OUTPUTDIM, Atype);
