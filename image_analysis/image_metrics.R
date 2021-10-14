# Michael Montalbano
library(reticulate)
library(SpatialVx)
library(MBCbook)
library(base)

# statistics on numbers
# performance metric p = sum(w*metric)

source_python("pickle_reader.py")
np <- import("numpy")


data <- data.frame(matrix(NA,nrow=940,ncol=22))
image_folder <- '~/data'

# trueouts = pickle_data$true_testing
# predouts = pickle_data$predict_testing

# trueouts = squeeze_it(trueouts)
# predouts = squeeze_it(predouts)

y_test = np$load('data/y_test_raw_noShear.npy')
y_pred = np$load('data/y_pred_raw_noShear.npy')
i = 93 # testing

for (i in 0:938) {
    print(i)
    true <- get_image(y_test,as.integer(i))
    pred <- get_image(y_pred,as.integer(i))
    df <- true
    df[df <=  as.double(18)] <- 0
    df[df > as.double(18)] <- 1

    df_pred <- pred
    df_pred[df_pred <=  as.double(18)] <- 0
    df_pred[df_pred > as.double(18)] <- 1   

    if ((sum(df) < 20))
    {
        print('too small')
        next 
    }
    if (sum(df_pred) < 20)
    {
        next
    }
    else {
    true[true < 15] = 0
    pred[pred < 15] = 0
    ?FeatureFinder
    hold <- make.SpatialVx(true,pred,field.type = "", units = "grid squares")
    look <- FeatureFinder(hold,min.size= 300)
    summary(look)
    intensityinfo <- summary(look)
    intensityinfo_X <- intensityinfo$X[c(1,2,3,4,5,6,7)]
    intensityinfo_X
    intensityinfo_Y <- intensityinfo$Y[c(4,5,6,7)]
    #intensity_info <- c() intensityinfo$X[1]

    # count number of objects in X
    res <- summary(look)
    xhat_n = 0
    for (obj in look$Y.feats)
    {
        xhat_n = xhat_n + 1
    }
    x_n = 0
    for (obj in look$X.feats)
    {
        x_n = x_n + 1
    }


    cent <- centmatch(look)
    res <- FeatureMatchAnalyzer(cent)
    info <- summary(res,silent=TRUE)

    if (is.null(info)) 
    {
        next
    }
    row_info <- info[1,c(1,2,3,4,8,9,10,11,14)]
    data[i + 1, 1] <- intensityinfo_X[1] # centroidX
    data[i + 1, 2] <- intensityinfo_X[2] # centroidY
    data[i + 1, 3] <- intensityinfo_X[3] # area
    data[i + 1, 4] <- intensityinfo_X[4] # orientation angle
    data[i + 1, 5] <- intensityinfo_X[5] # aspect ratio
    data[i + 1, 6] <- intensityinfo_X[6] # intensity 25th percentile
    data[i + 1, 7] <- intensityinfo_X[7] # intensity 90th percentile
    data[i + 1, 8] <- intensityinfo_Y[1] # orientation angle
    data[i + 1, 9] <- intensityinfo_Y[2] # orientation angle
    data[i + 1, 10] <- intensityinfo_Y[3] # intensity 25 Xhat
    data[i + 1, 11] <- intensityinfo_Y[4] # intensity 90 Xhat
    data[i + 1, 12] <- row_info[1] # medFalseAlarm
    data[i + 1, 13] <- row_info[2] # medMiss
    data[i + 1, 14] <- row_info[3] # msdFalsealarm
    data[i + 1, 15] <- row_info[4] # msdMiss
    data[i + 1, 16] <- row_info[5] # cent distance
    data[i + 1, 17] <- row_info[6] # angle diff
    data[i + 1, 18] <- row_info[7] # area ratio
    data[i + 1, 19] <- row_info[8] # int area
    data[i + 1, 20] <- row_info[9] # haus   
    data[i + 1, 21] <- x_n
    data[i + 1, 22] <- xhat_n
    }
}

write.csv(data,"image_metrics_raw_noShear.csv",row.names=FALSE,quote=FALSE)
data[1,1]

info

?FeatureMatcher


# record information about multiple objects per index. 
# index records with stormID or index (index record with index xD)
# threshold input images 
# record info about centroid placement
# see if centroid placement on the truth impacts predictablility 

# count number of NA cases
f=0
for (i in 0:938){
    print(i)

    true <- get_image(y_test,as.integer(i))
    df <- true
    df[df <=  as.double(20)] <- 0
    df[df > as.double(20)] <- 1
    sum(df)
    if (sum(df) > 15){
        f = f+1
        print('greater')
    }
}

cent <- centmatch(look)
res <- FeatureMatchAnalyzer(cent)

data <- data.frame(matrix(NA,nrow=400,ncol=1))
y_test = np$load('data/y_test_raw.npy')
y_pred = np$load('data/y_pred_raw.npy')
for (i in 1:400)
{
Z <- y_test[i,,]
Z[y_test[i,,] < 20] <- 0
imshow(Z)

X <- y_pred[i,,]
X[y_pred[i,,] < 20] <- 0
imshow(X)

summary <- Gbeta(Z,X,threshold=10,beta=5000000)
gbeta = Gbeta(Z,X,threshold=10,beta=5000000)
data[i,1] = gbeta
}

mean(data[0:400,1])
data_min <- data
data_raw <- data
data_noShear <- data
data_noMRMS <- data

data_raw - data_noMRMS

i = 389
Z <- y_test[i,,]
Z[y_test[i,,] < 20] <- 0
imshow(Z)

X <- y_pred[i,,]
X[y_pred[i,,] < 20] <- 0
imshow(X)

summary <- Gbeta(Z,X,threshold=10,beta=10000000)
summary[1]

