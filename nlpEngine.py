from nltk.tokenize import TreebankWordTokenizer as twt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class DocumentRanker:
    
    def __init__(self, documents):
        self.documents = documents
        # self.bert = BERTComponent('bert-large-cased')
        self.__model = SentenceTransformer('bert-large-cased')
    
    def __get_info_rep(self, document):
        pass
    
    def __get_embedding(self, text):
#         spans = twt().span_tokenize(text)
#         text_word_embeddings, text_embedding = self.bert.get_bert_embeddings(text, spans)
#         return text_embedding
        return self.__model.encode(text)
    
    def get_related_documents(self, query):
        index = {}
        last = 0
        related_documents = []
        
        q_sent_embedding = self.__get_embedding(query)
        
        for document in self.documents:
            abstract = document['documentTitle'] + " " + document['workAbstract'] + ". "
            for keyword in document['authorKeywords']:
                abstract += keyword['keyword']
            abstract_embedding = self.__get_embedding(abstract)
            #index[last] = torch.dot(q_sent_embedding, abstract_embedding)
            index[document['id']] = cosine_similarity([q_sent_embedding], [abstract_embedding])[0][0]
            # last += 1
        
        doc_scores = list(index.items())
        doc_scores = [(x[0], x[1].tolist()) for x in doc_scores]
        doc_scores= sorted(doc_scores, key = lambda x: x[1], reverse=True)
        #print("Scores: ", scores)
#         probs = F.softmax(scores, dim=0)
#         probs = [t.tolist() for t in probs]
#         probs.sort(reverse=True)
        #print("Probs: ", probs)
        # print(doc_scores)
        return doc_scores
        
        # if number_of_documents > len(doc_scores):
        #     return doc_scores
        # else:
        #     return doc_scores[:number_of_documents]