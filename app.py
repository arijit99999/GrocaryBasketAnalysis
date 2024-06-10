from flask import Flask, request, render_template
import pandas as pd
import os
import sys
from src.GrocaryBasketAnalysis.logger import logging
from src.GrocaryBasketAnalysis.exception import customexception
from src.GrocaryBasketAnalysis.utlis.utils import load_object
from mlxtend.frequent_patterns import apriori,association_rules

app = Flask(__name__,template_folder='template')

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def forecast():
        file = request.files['file']
        data=pd.read_csv(file)
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        d1=data.groupby('CustomerID')['Itemname'].apply(list).to_list()
        pre=load_object(os.path.join('artifacts',"encoder.pkl"))
        res1=pre.fit_transform(d1)
        d2=pd.DataFrame(res1,columns=pre.columns_)
        d2.replace(False,0,inplace=True)
        d2.replace(True,1,inplace=True)
        support=apriori(d2,min_support=.05,use_colnames=True)
        pred1=support.sort_values(by='support',ascending=False)
        pred1= pred1.to_html(classes='table table-striped') #convert data frame to a html
        confidence=association_rules(support,metric='confidence',min_threshold=.6)
        pred2=confidence.sort_values(by='confidence',ascending=False)
        pred2= pred2.to_html(classes='table table-striped')
        return render_template('result.html',pred1=pred1,pred2=pred2)
             
if __name__ == '__main__':
    app.run(debug=True)