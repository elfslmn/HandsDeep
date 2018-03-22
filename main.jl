using Knet, ImageView, JLD;
include("util.jl")

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

# Minibatch data
Atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}
dtst = minibatch(xtst,ytst,BATCHSIZE;xtype=Atype)
dtrn = minibatch(xtrn,ytrn,BATCHSIZE;xtype=Atype)
length(dtrn),length(dtst)
clear!(:xtrn); clear!(:ytrn); clear!(:xtst); clear!(:ytst);


#initilize baseline model parameter
w = initBase((240,320,1), 48, Atype);

trn_loss = Any[]
tst_loss = Any[]

function report(epoch, w)
    euc_trn = euc_loss_all(w,dtrn);
    push!(trn_loss, euc_trn)
    euc_tst = euc_loss_all(w,dtst)
    push!(tst_loss, euc_tst)
    println((:epoch,epoch,:trn,euc_trn,:tst,euc_tst))
end
report(0,w)

# ========== TRAINING ITERATIONS ============
@time for epoch=1:EPOCHS #
    train_sgd(w, dtrn, LR)
    report(epoch,w)
end
