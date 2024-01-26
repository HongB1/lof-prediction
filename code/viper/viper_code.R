getwd()
rm(list = ls())

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(version = "3.18")

# package
BiocManager::install("viper")
BiocManager::install('aracne.networks')
BiocManager::install('org.Hs.eg.db')
BiocManager::install("mixtools")
BiocManager::install("bcellViper")

# library
library(viper)
library(aracne.networks)
library(org.Hs.eg.db)
library(bcellViper)
library(mixtools)

#Regulon 불러오기 (coad 예시 데이터)
coad_regulon = regulonlaml
coad_regulon

# change to gene symbol
test = names(coad_regulon)
test[1:100]
gene_converting = AnnotationDbi::select(org.Hs.eg.db, keys = test, columns = "SYMBOL")
names(coad_regulon) = gene_converting$SYMBOL

for(i in 1:length(coad_regulon)){
  gene_name = AnnotationDbi::select(org.Hs.eg.db, keys = names(coad_regulon[[i]][[1]]), columns = "SYMBOL")  
  names(coad_regulon[[i]][[1]]) = gene_name[,2]}

# input data
input_path = '/home/hb/python/lof/perturbseq/gene_exp_for_viper/input'
file_list <- list.files(path = input_path, full.names = TRUE)

for (file_path in file_list){
  data <- read.csv(file_path, row.names=1, header=TRUE)
  res <- viper(data, coad_regulon)
  df <- as.data.frame(res)
  condition = strsplit(file_path, '/')[[1]][9]
  save_file_name = sprintf("/home/hb/python/lof/perturbseq/gene_exp_for_viper/output/%s", condition)
  write.csv(df, file=save_file_name)
  
  if (file_path=="/home/hb/python/lof/perturbseq/gene_exp_for_viper/input/CEBPB.csv"){
    break
  }
    
  
}

file_path
condition
data <- read.csv(file_path, row.names=1, header=TRUE)
head(data)

data(bcellViper, package="bcellViper")
d1 = exprs(dset)
#print(d1[1:10,1:10])

# gene expression을 input으로 viper 돌리기
res <- viper(data, coad_regulon)
df <- as.data.frame(res)
write.csv(df, file="/home/hb/python/lof/perturbseq/gene_exp_for_viper/norman_cebpb_viper_result.csv")
res['CEBPB']
print(res[1:5, 1:5])
print(dim(d1)); print(dim(res))

