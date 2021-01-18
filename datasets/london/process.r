library(easypackages)
libraries("spdep", "maptools")

ELmap <- readShapePoly("/home/junteng/Downloads/London/statistical-gis-boundaries-london/ESRI/London_Ward")
ELnb <- poly2nb(ELmap, queen=T)
lw = nb2listw(ELnb, glist=NULL, zero.policy=NULL)
W = as(as_dgRMatrix_listw(nb2listw(ELnb)), "CsparseMatrix")

write.csv(as.data.frame(ELmap$GSS_CODE), "code.csv")
write.csv(as.data.frame(as.matrix(W)), "W.csv")
