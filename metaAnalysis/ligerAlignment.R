library(liger)

# wd "datasets"
setwd(paste("/Users/calebreagor/",
            "Library/Mobile\ ",
            "Documents/com~apple",
            "~CloudDocs/Documents/",
            "grad/HudspethLab/",
            "metaAnalysis/datasets/", sep=""))


# load single cell datasets -> liger object
lush=read.table("lush.csv",
                header=TRUE,
                sep=',',row.names=1)

burns=read.table("burns.csv",
                 header=TRUE,
                 sep=',',row.names=1)

hoa=read.table("hoa.csv",
               header=TRUE,
               sep=',',row.names=1)

data.data <- list(lush_data=lush,
                  burns_data=burns,
                  hoa_data=hoa)

liger.data <- createLiger(data.data)


# preprocess: normalize, find variable genes, scale
liger.data <- normalize(liger.data)
liger.data <- selectGenes(liger.data,
                          do.plot=F)
liger.data <- scaleNotCenter(liger.data)


# factorization and quantile alignment
k.suggest <- 20
liger.data <- optimizeALS(liger.data,
                          k=k.suggest,
                          thresh=5e-5,
                          nrep=3)

liger.data <- quantile_norm(liger.data,
                            do.center=T)


# write output files
fwrite(x=liger.data@H.norm,
       row.names=TRUE,
       file="liger_H.norm.csv")
