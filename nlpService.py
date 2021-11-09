from flask import Flask
from flask_restful import Resource, Api, reqparse
from nlpEngine import *
import pandas as pd
import ast


app = Flask(__name__)
api = Api(app)

# class ResponseDTO:
    


class NLP(Resources):
    # def get(self):
    #     data = pd.read_csv('users.csv')  # read CSV
    #     data = data.to_dict()  # convert dataframe to dictionary
    #     return {'data': data}, 200  # return data and 200 OK code
    
    def post(self):
        parser = reqparse.RequestParser()  # initialize
        
        parser.add_argument('query', required=True)  # add args
        parser.add_argument('documents', required=True)
        # parser.add_argument('city', required=True)
        
        args = parser.parse_args()  # parse arguments to dictionary
        query=args['query'] 
        documents = args['articles']
        
        ranker = DocumentRanker(documents)
        results = ranker.get_related_documents(query)
        
        
        # result = pd.DataFrame({
        #     'userId': args['userId'],
        #     'name': args['name'],
        #     'city': args['city'],
        #     'locations': [[]]
        # })
        
        return {'data': results}, 200  # return data with 200 OK


api.add_resource(NLP, '/nlp')  # add endpoints



if __name__ == '__main__':
    app.run()  # run our Flask app
