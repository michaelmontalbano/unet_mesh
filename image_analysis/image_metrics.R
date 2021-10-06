# Michael Montalbano
library("reticulate")
library(SpatialVx)
source_python("pickle_reader.py")
np <- import("numpy")

data <- as.data.frame(data)

image_folder <- '~/data'


for (i in 1:939){

    # paste

    true <- np$load("sample{}.npy")
    pred <- np$load("sample5_pred+{}.npy")

    hold <- make.SpatialVx(true2,pred2,thresholds = 10,,field.type = "", units = "grid squares")
    look <- FeatureFinder(hold)
    summary(look)
    cent <- centmatch(look)
    res <- FeatureMatchAnalyzer(cent)
    infomin <- summary(res)
    row <- infomin[1,c(1,2,3,4,9,10,11)]
    data[i,] <- row
}

