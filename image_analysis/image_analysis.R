library(reticulate)
library(SpatialVx)
library(MBCbook)

version <- "3.8.10"
install_python(version = version)
virtualenv_create("new", python_version = version)
use_virtualenv("new", required = TRUE)

pd <- import("pandas")
np <- import("numpy")
os <- import("os")

pickle_data <- pd$read_pickle("/home/michaelm/results/standardScalar_guass_0.1_unet_mse_multi_dropout_None_batchsize_50_concat_l2_0.1_raw_opt_adam_regression_results.pkl")


source_python("pickle_reader.py")
library(map)
# enable the [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/en/master/index.html) [`auto_suggest` feature](https://python-prompt-toolkit.readthedocs.io/en/master/pages/asking_for_input.html#auto-suggestion)
# this option is experimental and is known to break python prompt, use it with caution
pd <- import('pandas')
library('abind')

true_outs <- read_pickle_file("/home/michaelm/results/standardScalar_guass_0.1_unet_mse_multi_dropout_None_batchsize_50_concat_l2_0.1_raw_opt_adam_regression_results.pkl",pred='False')
pred_outs <- read_pickle_file("/home/michaelm/results/standardScalar_guass_0.1_unet_mse_multi_dropout_None_batchsize_50_concat_l2_0.1_raw_opt_adam_regression_results.pkl",pred='True')

dim(pred_outs)
array <- 1:10
2+3
pickle_data <- pd$read_pickle("/home/michaelm/results/standardScalar_guass_0.1_unet_mse_multi_dropout_None_batchsize_50_concat_l2_0.1_raw_opt_adam_regression_results.pkl")
names(pickle_data)
true_outs = pickle_data$true_outs
install.packages('map')
np <- import("numpy")
mat <- np$load("true_outs.npy")
mat
dim(mat)

true_outs = np$load("true_outs.npy")
true_testing = np$load("true_testing.npy")

true_outs[0,1,1,1]

true = get_image(true_outs,index=10)

?hoods2d
hoods2d(hold)
hist(hold)

look <- hoods2d(hold, which.methods=c("multi.event", "fss"), 
    levels=c(1, 3, 5, 9, 17), verbose=TRUE)


true_outs <- pickle_data$true_testing
pred_outs <- pickle_data$predict_testing

true <- mat[0,,]

true <- np$load("sample.npy")
plot(true)
pred <- np$load("sample1_pred.npy")
?make.SpatialVx
hold <- make.SpatialVx(true,pred,thresholds = 50,field.type = "", units = "grid squares")
hist(hold)
look <- FeatureFinder(hold)
cent <- centmatch(look)
summary(cent)
res <- FeatureMatchAnalyzer(cent)
summary(look)
info <- summary(res)
info[1,c(1,2,3,4,9,10,11)]

true1_Min = np$load('sample_Min.npy')
pred1_min = np$load("sample_Min_pred.npy")
true2_Min = np$load('sample2_Min.npy')
pred2_Min = np$load("sample2_Min_pred.npy")

pred <- np$load('sample1_pred.npy')
true <- np$load('sample1.npy')
true2 <- np$load("sample2.npy")
pred2  <- np$load("sample2_pred.npy")
true3 <- np$load("sample3.npy")
pred3 <- np$load("sample3_pred.npy")
true4 <- np$load("sample5.npy")
pred4 <- np$load("sample5_pred.npy")
imshow(pred)
imshow(true)

imshow(pred3)
colorbar.plot(pred2,)
?make.SpatialVx
hold <- make.SpatialVx(pred,true,thresholds = 30,field.type = "random", units = "grid squares")
look <- FeatureFinder(hold)
summary <- summary(look)

hold2 <- make.SpatialVx(true2,pred2,thresholds = 10, units = "grid squares")
plot(hold2)
hold3 <- make.SpatialVx(true3,pred3,thresholds = 10,field.type = "random", units = "grid squares")

?centmatch
cent <- centmatch(look)
summary(cent)
int <- interester(cent)

look2 <- deltamm(look, N= 600, verbose = TRUE)

# try 2
res <- FeatureMatchAnalyzer(cent)
info <- summary(res)
info[1,c(1,2,3,4,9,10,11)]

true <- true4
pred <- true4

# true <- true4
# pred <- pred4

hold <- make.SpatialVx(true2,pred2,thresholds = 10,,field.type = "", units = "grid squares")
look <- FeatureFinder(hold) 
summary(look)
cent <- centmatch(look)
res <- FeatureMatchAnalyzer(cent)
res
infomin <- summary(res)
info
infomin[1,c(1,2,3,4,9,10,11)]
info
info2Min
# contingency table time
tres <- FeatureTable(cent)
summary(tres)

# change featurefinder
look <- FeatureFinder(hold3, thresh=10,min.size=4)
ang1 <- FeatureAxis(look$X.feats[[1]])
ang2 <- FeatureAxis(look$Y.feats[[1]])
plot(ang1)
plot(ang2)
ang2
?locmeasures2d
locmeasures2d(hold,k=c(1,3))
locmeasures2d(hold3,k=c(1,3))

Gbeta(X=true,Xhat=pred,threshold=20, beta=0.2)
Gbeta(X=true3,Xhat=pred3,threshold=10, beta=601*501/2)
TheBigG(true,pred,20)
TheBigG(true2,pred2,10)
TheBigG(true3,pred3,10)

Gbeta(true,pred,threshold=20,beta=3900)
TheBigG(true1_Min,pred1_min,threshold=20,beta=3900)


summary <- TheBigG()

Gbeta(true2_Min,pred2_Min,threshold=10,beta=601*501/2)
TheBigG(true2_Min,pred2_Min,threshold=10,beta=601*501/2)
TheBigG(true4,pred4,threshold=10,beta=601*501/2)
