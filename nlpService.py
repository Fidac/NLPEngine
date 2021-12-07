from transformers import BertTokenizer, BertModel, BertForMaskedLM, XLNetConfig, XLNetModel, XLNetTokenizer
import torch
from torch.nn.functional import one_hot
from nltk.tokenize import TreebankWordTokenizer as twt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask import request
import pandas as pd
import ast
import json
import numpy as np


class BERTComponent:
    tokenizer = None
    bert_model = None

    def __init__(self, model):
        self.bert_vector_size = 3072
        self.sent_vector_size = 768
        self.model = model
        print("Tokenizer: ", BERTComponent.tokenizer)
        self.tokenizer = BERTComponent.tokenizer if BERTComponent.tokenizer else BertTokenizer.from_pretrained(model)
        BERTComponent.tokenizer = self.tokenizer
        self.bert_model = BERTComponent.bert_model if BERTComponent.bert_model else BertModel.from_pretrained(model)
        BERTComponent.bert_model = self.bert_model
        self.bert_model.eval()


    def get_bert_spans(self, words, bert_tokens):
        if self.model == 'bert-large-uncased':
            words = [self._flat_word(word) for word in words]

        i = 0
        j = 1
        idx = 0

        bert_words_indexes = []
        bert_words = []
        while i < len(words):
            word = words[i]

            bert_word = bert_tokens[j]
            bert_word = bert_word[2:] if bert_word.startswith("##") else bert_word
            bert_word = bert_word[idx:]

            #Spacing control
            if word in [" ", "  ", "   "]:
                bert_words.append([word])
                bert_words_indexes.append([-1])

            #When the current word is [UNK] for bert
            elif bert_word == "[UNK]":
                bert_words.append(["[UNK]"])
                bert_words_indexes.append([j])
                j += 1
                idx = 0

            #When the current word is contained in bert token. Very weird
            elif len(word) < len(bert_word) and bert_word.find(word) >= 0:
                bert_words.append([bert_word])
                bert_words_indexes.append([j])

                idx = bert_word.find(word) + len(word)
                if idx == len(bert_word):
                    j += 1
                    idx = 0

            #Otherwise
            else:
                k = 0
                span = []
                span_indexes = []

                while k < len(word):
                    if word.find(bert_word, k) == k:
                        span.append(bert_word)
                        span_indexes.append(j)
                        k += len(bert_word)
                        j += 1
                        idx = 0
                        bert_word = bert_tokens[j]
                        bert_word = bert_word[2:] if bert_word.startswith("##") else bert_word
                    else:
                        print("Error")
                        return bert_words, bert_words_indexes

                bert_words.append(span)
                bert_words_indexes.append(span_indexes)

            i += 1

        assert len(bert_words_indexes) == len(words)

        return bert_words, bert_words_indexes

    def _flat_word(self, word):
        word = word.lower()
        word = word.replace("ñ", "n")
        word = word.replace("á", "a")
        word = word.replace("é", "e")
        word = word.replace("í", "i")
        word = word.replace("ó", "o")
        word = word.replace("ú", "u")
        word = word.replace("ä", "a")
        word = word.replace("ü", "u")
        word = word.replace("ö", "o")
        word = word.replace("ū", "u")
        word = word.replace("ā", "a")
        word = word.replace("ī", "i")
        word = word.replace("ș", "s")
        word = word.replace("ã", "a")
        word = word.replace("ô", "o")

        return word

    def _sum_merge(self, vectors):
        return torch.sum(torch.stack(vectors), dim=0)

    def _mean_merge(self, vectors):
        return torch.mean(torch.stack(vectors), dim=0)

    def _last_merge(self, vectors):
        return vectors[-1]

    def _get_merge_tensors(self, token_vec_sums, words_indexes):
        pad_tensor = torch.zeros(self.bert_vector_size)
        real_vec = []
        for word_indexes in words_indexes:
            vectors = [(token_vec_sums[idx] if idx != -1 else pad_tensor) for idx in word_indexes]
            real_vec.append(self._mean_merge(vectors))

        return real_vec

    def get_bert_embeddings(self, sentence, spans):
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        segments_ids = [1] * len(tokenized_sentence)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            encoded_layers = self.bert_model(tokens_tensor, segments_tensors, output_hidden_states=True)

        #print("This is enconded layers: ", len(encoded_layers.hidden_states))
        
        token_embeddings = torch.stack(encoded_layers.hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vec_sums = []
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=-1)
            token_vec_sums.append(cat_vec)

        words = [sentence[beg:end] for (beg, end) in spans]
        bert_words, bert_words_indexes = self.get_bert_spans(words, tokenized_sentence)

        bert_embeddings = self._get_merge_tensors(token_vec_sums, bert_words_indexes)
        sentence_embedding = torch.mean(torch.stack(token_vec_sums), dim=0)
        
        return bert_embeddings, sentence_embedding
class DocumentRanker:
    
    def __init__(self, documents):
        self.documents = documents
#         self.bert = BERTComponent('bert-large-cased')
        self.__model = SentenceTransformer('bert-large-cased')
    
    def __get_info_rep(self, document):
        pass
    
    def __get_embedding(self, text):
#         spans = twt().span_tokenize(text)
#         text_word_embeddings, text_embedding = self.bert.get_bert_embeddings(text, spans)
#         return text_embedding
        return self.__model.encode(text)
    
    def get_clustered_documents(self):
        
        clusters = {}
        centroids = {}
        clusterNumber = 1
        clusters[clusterNumber] = [(self.documents[0]['id'],self.documents[0]['embedding'])]
        centroids[clusterNumber] = self.documents[0]['embedding']
        
        for i in range(1, len(self.documents)):
            document = self.documents[i]
            inserted = False
            for j in range(1, clusterNumber + 1):
                centroid = centroids[clusterNumber]
                score = cosine_similarity([centroid], [document['embedding']])[0][0]
                #print("SCORE: ", score)
                if score > 0.9:
                    inserted = True
                    clusters[j].append((document['id'], document['embedding']))
                    centroids[j] = document['embedding']
                    break
            
            if(not inserted):
                clusterNumber += 1
                clusters[clusterNumber] = [(document['id'],document['embedding'])]
                centroids[clusterNumber] = document['embedding']
        
        return list(clusters.items())
    
    def get_related_documents(self, query):
        index = {}
        embeddings = {}
        last = 0
        related_documents = []
        
        q_sent_embedding = self.__get_embedding(query)
        
        for document in self.documents:
            abstract = document['title'] + " " + document['abstractText'] + ". "
#             for keyword in document['authorKeywords']:
#                 abstract += keyword['keyword']
            if(document['embedding'] is not None and len(document['embedding']) > 0):
                #print("USING SAVED EMBEEDING")
                #abstract_embedding = torch.FloatTensor(document['embedding'])
                abstract_embedding = np.asarray(document['embedding'], dtype=np.float32)
            else:
                #print("Computing new Embedding")
                abstract_embedding = self.__get_embedding(abstract)
                document['embedding'] = abstract_embedding
            #print("This is the embedding type: ", type(abstract_embedding))
            #print("This is the embedding: ", abstract_embedding)
            #index[last] = torch.dot(q_sent_embedding, abstract_embedding)
            index[document['id']] = (cosine_similarity([q_sent_embedding], [abstract_embedding])[0][0], abstract_embedding.tolist())
#             embeddings[document['id']] = abstract_embedding.toList()
#             last += 1
        
        doc_scores = list(index.items())
        doc_scores = [(x[0], x[1][0].tolist(), x[1][1]) for x in doc_scores]
        doc_scores= sorted(doc_scores, key = lambda x: x[1], reverse=True)
        return doc_scores
        #print("Scores: ", scores)
#         probs = F.softmax(scores, dim=0)
#         probs = [t.tolist() for t in probs]
#         probs.sort(reverse=True)
        #print("Probs: ", probs)
        # print(doc_scores)
        
#         if number_of_documents > len(doc_scores):
#             return doc_scores
#         else:
#             return doc_scores[:number_of_documents]
        
        

class NLP(Resource):
    # def get(self):
    #     data = pd.read_csv('users.csv')  # read CSV
    #     data = data.to_dict()  # convert dataframe to dictionary
    #     return {'data': data}, 200  # return data and 200 OK code

    def post(self):        
        #args = parser.parse_args()  # parse arguments to dictionary
        args = request.get_json()
        #args = request.args.get('query')
        #print(args)
        calculateClusters = args['calculateClusters']
        query=args['query']
        #print(query)
#         args['articles'] = args['articles'].replace("\'", "\"")
        #print(args['articles'])
        documents = args['articles']

        ranker = DocumentRanker(documents)
        results = ranker.get_related_documents(query)

        if(calculateClusters):
            clusterResults = ranker.get_clustered_documents()

        response = []

        if(calculateClusters):
            for result in results:
                clusterNumber = 0
                for clusterResult in clusterResults:
                    #print("CCLCLCLC:", clusterResult)
                    #print("AAAAAA: ",clusterResult[1], result[0])
                    #print("BBBBBB:", [x[1] for x in clusterResult])
                    if(result[0] in [x[0] for x in clusterResult[1]]):
                        clusterNumber = clusterResult[0]
                        break

                response.append({"id": result[0], "weight": result[1], "embedding": result[2], "clusterId": clusterNumber})

        else:
            for result in results:
                response.append({"id": result[0], "weight": result[1], "embedding": result[2]})

        # result = pd.DataFrame({
        #     'userId': args['userId'],
        #     'name': args['name'],
        #     'city': args['city'],
        #     'locations': [[]]
        # })

        #print("RESPONSE: ", response)

        return response, 200  # return data with 200 OK


if __name__ == '__main__':
    
    app = Flask("NlpApp")
    api = Api(app)
    api.add_resource(NLP, '/nlp')  # add endpoints
    app.run(host='0.0.0.0', port='5000', debug=True)  # run our Flask app
