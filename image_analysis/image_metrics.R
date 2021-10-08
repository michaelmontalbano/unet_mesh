# Michael Montalbano
# For use in image verification 
# Requires pickle_reader.py to initialize environement, open pickle, and manipulate data
library(reticulate)
library(SpatialVx)
library(MBCbook)
library(base)

# statistics on numbers
# performance metric p = sum(w*metric)

source_python("pickle_reader.py")
np <- import("numpy")


data <- data.frame(matrix(NA,nrow=940,ncol=17))
image_folder <- '~/data'

# trueouts = pickle_data$true_testing
# predouts = pickle_data$predict_testing

# trueouts = squeeze_it(trueouts)
# predouts = squeeze_it(predouts)

y_test = np$load('data/y_test_raw_min.npy')
y_pred = np$load('data/y_pred_raw_min.npy')
n = 601
true <- get_image(y_test,as.integer(n))
pred <- get_image(y_pred,as.integer(n))
imshow(true)
max(pred)

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
    hold <- make.SpatialVx(true,pred,units = "grid squares")
    look <- FeatureFinder(hold,min.size= 300)
    intensityinfo <- summary(look)
    intensityinfo_X <- intensityinfo$X[c(1,2,3,4,5,6,7)]
    intensityinfo_Y <- intensityinfo$Y[c(4,5,6,7)]
    #intensity_info <- c() intensityinfo$X[1]

    # count number of objects in X
    res <- summary(look)
    m = 0
    for (obj in look$Y.feats)
    {
        xhat_n = m + 1
    }
    m = 0
    for (obj in look$X.feats)
    {
        x_n = m + 1
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

write.csv(data,"image_metrics_raw_min.csv",row.names=FALSE,quote=FALSE)
