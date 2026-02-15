for the target identification i used the open source pretrained dylan/ indix clm roberta model which reuslted in 71 clustered targets but it didnt seem like correct as there was kp oli and kp chor as two different cllusters so i have to change the code and use the new model.

similary for the devanagari pipeline, we have used static fasttext embeddings which is not that good so in the paper we referred i.e. situala and shahi they are using custom trained fasttext. if we use custom trained model, then the confidence score will also be more than the generic ones.

at first for the target identification and normalization, we used the sentence transformer embedidings and agglomerative clustering which is good for merging similar targets and choosing a cannonical form based on frequence but due to low similarity threshold 0.35, we got reduncdant clusters. this occured due to lack of lemmatization as well cause nepal, nepalma and nepalko were treated differently.

we have not considered mmulti world target as nepal government, which can split into two parts as nepal and government differently.

there is no semantic normalization beyond clustering

Improvements that can be done:::
1) multi word entity recognition

2) text normalization before NER
Optional: apply Nepali-specific Unicode normalization (unicodedata.normalize('NFC', text)).
 there is another issue in our yotube comments as 
 if the same comment or the word is typed from windows or from mac/linus they will have different unicode embedding and they will be stored in the vocabulary resulting in the duplication. so to solve this problem we can use unicode normalization.

 3) we can use better embeddings and consider the dynamic clustering threshold.


 the clear steps that i have to follow now:::
 1. normalize the text for unicode and whitespace before ner
 2. multi word entity aggregation carefully.
 3. apply morphological normalization or lemmatization
 4. change the embedding to use mean pooled embeddings for better context
 5. dynamic clustering DBSCAN with cosine distance ot tune threshould using sihouette score.
 6. combine frequency + embedding centrality.
 7. build dropdown.
 