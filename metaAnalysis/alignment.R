#####################################################################################################################################
############################################## ALIGN DATASETS #######################################################################
#####################################################################################################################################

# dependencies
library(Seurat)
library(data.table)

# load cc genes from wd "refs"
setwd(paste("/Users/calebreagor/",
            "Library/Mobile\ ",
            "Documents/com~apple",
            "~CloudDocs/Documents/",
            "grad/HudspethLab/",
            "metaAnalysis/refs/",
            sep=""))

s_genes=read.table("s_phase_genes.csv",
                   header=TRUE,
                   sep=',',
                   row.names=1)

g2m_genes=read.table("g2m_phase_genes.csv",
                     header=TRUE,
                     sep=',',
                     row.names=1)


# wd "datasets"
setwd(paste("/Users/calebreagor/",
            "Library/Mobile\ ",
            "Documents/com~apple",
            "~CloudDocs/Documents/",
            "grad/HudspethLab/",
            "metaAnalysis/datasets/", 
            sep=""))


# load single cell datasets
lush=read.table("lush.csv",
                header=TRUE,
                sep=',',
                row.names=1)

burns=read.table("burns.csv",
                 header=TRUE,
                 sep=',',
                 row.names=1)

hoa=read.table("hoa.csv",
               header=TRUE,
               sep=',',
               row.names=1)

lush_seur <- CreateSeuratObject(counts=lush)
burns_seur <- CreateSeuratObject(counts=burns)
hoa_seur <- CreateSeuratObject(counts=hoa)


# preprocessing
n_features <- 8500

lush_seur <- NormalizeData(lush_seur)
burns_seur <- NormalizeData(burns_seur)
hoa_seur <- NormalizeData(hoa_seur)

lush_seur <- FindVariableFeatures(lush_seur,
                                  selection.method="vst",
                                  nfeatures=n_features)

burns_seur <- FindVariableFeatures(burns_seur,
                                   selection.method="vst",
                                   nfeatures=n_features)

hoa_seur <- FindVariableFeatures(hoa_seur,
                                 selection.method="vst",
                                 nfeatures=n_features)


# assign cells to cell cycle
lush_seur <- CellCycleScoring(lush_seur,
                              set.ident=TRUE,
                              s.features=s_genes$Gene.name,
                              g2m.features=g2m_genes$Gene.name)

burns_seur <- CellCycleScoring(burns_seur,
                               set.ident=TRUE,
                               s.features=s_genes$Gene.name,
                               g2m.features=g2m_genes$Gene.name)

hoa_seur <- CellCycleScoring(hoa_seur,
                             set.ident=TRUE,
                             s.features=s_genes$Gene.name,
                             g2m.features=g2m_genes$Gene.name)


# align datasets
n_anchors <- 50

anchors <- FindIntegrationAnchors(c(lush_seur,
                                    burns_seur,
                                    hoa_seur),
                                  dims=1:n_anchors)

integrated <- IntegrateData(anchorset=anchors,
                            dims=1:n_anchors,
                            features.to.integrate=rownames(lush),)

DefaultAssay(integrated) <- "integrated"
integrated <- ScaleData(integrated)


# write output files
data_out <- as.data.frame(GetAssayData(object=integrated,
                                       assay="integrated",
                                       slot="data"))
fwrite(x=data_out,
       row.names=TRUE,
       file="aligned.csv")

phases_out <- as.data.frame(integrated@meta.data$Phase)
fwrite(x=phases_out,
       row.names=TRUE,
       file="phases.csv")

features_out <- as.data.frame(anchors@anchor.features)
fwrite(x=features_out,
       row.names=TRUE,
       file="features.csv")
