import json, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore', category=ConvergenceWarning)
RANDOM_STATE=17342; RESULTS_DIR=Path('results'); RESULTS_DIR.mkdir(exist_ok=True)
df=pd.read_csv('Grisoni_et_al_2016_EnvInt88.csv').dropna(); y=df['Class'].astype(int)
X=df.drop(columns=['CAS','SMILES','Set','logBCF','Class'], errors='ignore'); features=list(X.columns)
models={
 'Linear Classifier': Perceptron(max_iter=1000, random_state=RANDOM_STATE),
 'Logistic Regression': LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
 'KNN': KNeighborsClassifier(n_neighbors=5),
 'Gaussian NB': GaussianNB(),
 'Neural Network': MLPClassifier(hidden_layer_sizes=(8,), max_iter=60, random_state=RANDOM_STATE, early_stopping=True),
}
def eval_cv(model, Xsub, mode='final'):
    cv=RepeatedKFold(n_splits=10,n_repeats=10,random_state=RANDOM_STATE) if mode=='final' else KFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
    return tuple(map(float, (lambda s:(s.mean(),s.std()))(cross_val_score(Pipeline([('scaler',StandardScaler()),('model',model)]),Xsub,y,cv=cv,scoring='accuracy',n_jobs=1))))
part2_rows=[]
for n,m in models.items():
    mean,std=eval_cv(m,X,mode='final')
    part2_rows.append({'Algorithm':n,'Mean Accuracy':mean,'Std':std})
part2=pd.DataFrame(part2_rows)
part2.to_csv(RESULTS_DIR/'part2_algorithm_comparison.csv',index=False)
part3=[]; hist=[]
for name,model in models.items():
    selected=[]; remaining=features.copy(); current=0
    while remaining:
        best_candidate=None
        for f in remaining:
            cols=selected+[f]; mean,std=eval_cv(model,X[cols],mode='select')
            hist.append({'Algorithm':name,'Step':len(cols),'Candidate Added':f,'Features':', '.join(cols),'Selection Mean Accuracy':mean,'Selection Std':std})
            if best_candidate is None or mean>best_candidate[0] or (np.isclose(mean,best_candidate[0]) and std<best_candidate[1]): best_candidate=(mean,std,f,cols)
        if best_candidate[0] > current + 1e-6:
            current=best_candidate[0]; selected=best_candidate[3]; remaining.remove(best_candidate[2])
        else: break
    mean,std=eval_cv(model,X[selected],mode='final')
    print(name, selected, mean, std, flush=True)
    part3.append({'Algorithm':name,'Best Feature Subset':', '.join(selected),'Number of Features':len(selected),'Selection Mean Accuracy':current,'Selection Std':best_candidate[1] if 'best_candidate' in locals() else None,'Mean Accuracy':mean,'Std':std})
part3=pd.DataFrame(part3); part3.to_csv(RESULTS_DIR/'part3_feature_selection.csv',index=False); pd.DataFrame(hist).to_csv(RESULTS_DIR/'feature_search_history.csv',index=False)
class_dist=y.value_counts().sort_index(); class_dist.to_csv(RESULTS_DIR/'class_distribution.csv',header=['count'])
plt.figure(figsize=(6,4)); class_dist.plot(kind='bar'); plt.title('Class Distribution'); plt.xlabel('Class'); plt.ylabel('Count'); plt.tight_layout(); plt.savefig(RESULTS_DIR/'class_distribution.png',dpi=150); plt.close()
plt.figure(figsize=(8,4.5)); plt.bar(part2['Algorithm'],part2['Mean Accuracy'],yerr=part2['Std'],capsize=4); plt.title('Part 2 Accuracy with All Features'); plt.ylabel('Accuracy'); plt.ylim(0,1); plt.xticks(rotation=30,ha='right'); plt.tight_layout(); plt.savefig(RESULTS_DIR/'part2_accuracy.png',dpi=150); plt.close()
comp=part2[['Algorithm','Mean Accuracy']].merge(part3[['Algorithm','Mean Accuracy']],on='Algorithm',suffixes=(' All Features',' Selected Features'))
x=np.arange(len(comp)); width=.35
plt.figure(figsize=(8,4.5)); plt.bar(x-width/2,comp['Mean Accuracy All Features'],width,label='All features'); plt.bar(x+width/2,comp['Mean Accuracy Selected Features'],width,label='Selected features'); plt.title('All Features vs Selected Features'); plt.ylabel('Accuracy'); plt.ylim(0,1); plt.xticks(x,comp['Algorithm'],rotation=30,ha='right'); plt.legend(); plt.tight_layout(); plt.savefig(RESULTS_DIR/'feature_selection_comparison.png',dpi=150); plt.close()
with open(RESULTS_DIR/'metadata.json','w') as f: json.dump({'dataset':'QSAR Bioconcentration Classes','original_shape':[779,14],'post_dropna_shape':list(df.shape),'class_distribution':{str(k):int(v) for k,v in class_dist.items()},'features_used':features,'dropped_columns':['CAS','SMILES','Set','logBCF'],'target':'Class','random_state':RANDOM_STATE,'part2_cv':'RepeatedKFold(n_splits=10, n_repeats=10), reduced from 100 repeats due runtime','part3_method':'greedy forward selection','part3_selection_cv':'KFold(n_splits=5, shuffle=True)','part3_final_cv':'RepeatedKFold(n_splits=10, n_repeats=10)'},f,indent=2)
