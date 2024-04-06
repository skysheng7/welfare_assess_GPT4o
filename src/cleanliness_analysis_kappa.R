setwd("~/Desktop/vetmeduni/GPT4V_Welfare")

library(irr)


#Load the chatgpt cleanliness assessment results
res=read.csv("welfare_assess_cleanliness_result2.csv", sep = ",",header = T)

orig_res <- res[which(res$treatment=="original"),1:6]
seg_res <- res[which(res$treatment=="segment"),1:6]
segbp_res <- res[which(res$treatment=="segment_bodyPart"),1:6]
segbp_res <- segbp_res[-which(segbp_res$predict_score=="unable_assess"),]

#Kappa for each method and body part separately

#Udder
kappa2(orig_res[which(orig_res$assess_area=="udder cleanliness"),c(4,6)], weight = "unweighted")
kappa2(seg_res[which(seg_res$assess_area=="udder cleanliness"),c(4,6)], weight = "unweighted")
kappa2(segbp_res[which(segbp_res$assess_area=="udder cleanliness"),c(4,6)], weight = "unweighted")

#Hindleg
kappa2(orig_res[which(orig_res$assess_area=="hindleg cleanliness"),c(4,6)], weight = "unweighted")
kappa2(seg_res[which(seg_res$assess_area=="hindleg cleanliness"),c(4,6)], weight = "unweighted")
kappa2(segbp_res[which(segbp_res$assess_area=="hindleg cleanliness"),c(4,6)], weight = "unweighted")

#Hindquarter
kappa2(orig_res[which(orig_res$assess_area=="hindquarter cleanliness"),c(4,6)], weight = "unweighted")
kappa2(seg_res[which(seg_res$assess_area=="hindquarter cleanliness"),c(4,6)], weight = "unweighted")
kappa2(segbp_res[which(segbp_res$assess_area=="hindquarter cleanliness"),c(4,6)], weight = "unweighted")

#All
kappa2(orig_res[,c(4,6)], weight = "unweighted")
kappa2(seg_res[,c(4,6)], weight = "unweighted")
kappa2(segbp_res[,c(4,6)], weight = "unweighted")

#Hindquarter+Hindleg
kappa2(orig_res[-which(orig_res$assess_area=="udder cleanliness"),c(4,6)], weight = "unweighted")
kappa2(seg_res[-which(seg_res$assess_area=="udder cleanliness"),c(4,6)], weight = "unweighted")
kappa2(segbp_res[-which(segbp_res$assess_area=="udder cleanliness"),c(4,6)], weight = "unweighted")
