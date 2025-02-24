from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

def myEvaulate2(yture, ypred):
    precision=precision_score(yture,ypred)
    recall=recall_score(yture,ypred)
    f1=f1_score(yture,ypred)
    mcc=matthews_corrcoef(yture,ypred)
    acc=accuracy_score(yture,ypred)
    return precision,recall,f1,mcc,acc

def myEvaulate(yture, ypred):
    precision=precision_score(yture,ypred)
    recall=recall_score(yture,ypred)
    f1=f1_score(yture,ypred)
    mcc=matthews_corrcoef(yture,ypred)
    return precision,recall,f1,mcc