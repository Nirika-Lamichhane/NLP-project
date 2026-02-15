for the target identification i used the open source pretrained dylan/ indix clm roberta model which reuslted in 71 clustered targets but it didnt seem like correct as there was kp oli and kp chor as two different cllusters so i have to change the code and use the new model.

similary for the devanagari pipeline, we have used static fasttext embeddings 

at first for the target identification and normalization, we used the sentence transformer embedidings and agglomerative clustering which is good for merging similar targets and choosing a cannonical form based on frequence but due to low similarity threshold 0.35, we got reduncdant clusters. this occured due to lack of lemmatization as well cause nepal, nepalma and nepalko were treated differently.