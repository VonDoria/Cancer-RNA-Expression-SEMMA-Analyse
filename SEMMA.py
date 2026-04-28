#%%
import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
from feature_engine import discretisation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# Aumentando o limite de visualização do pandas para o caso em questão
# pd.options.display.max_columns = 21000
# pd.options.display.max_rows = 801
#%%



# ORIGEM DOS DADOS => https://www.kaggle.com/datasets/waalbannyantudre/gene-expression-cancer-rna-seq-donated-on-682016?resource=download
df = pd.read_csv('data.csv')
df_labels = pd.read_csv('labels.csv')
df['Result'] = df_labels['Class']
df.head()
#%%



df.shape #Mostra o numero de linhas e colunas
df.info() #Mostra um resumo do DataFrame
df.describe() #Gera tabela exploratória para todas as colunas
#%%

# DIVIDIR FEATURES E TARGET, X E Y

features = df.columns[1:-1]
target = 'Result'
X, y = df[features], df[target]
#%%

# ============================SAMPLE=============================
# ['Train/Test', 'OutOfTime', 'Balanceamento', 'Filtros']


# SEPARANDO DADOS DE TREINO(80%) E TESTE(20%)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
#%%

# COMPARANDO A HOMOGENIEDADE DOS GRUPOS DE TREINO E TESTE (BALANCEAMENTO = 'stratify=y')
print("Original: " + str(y.value_counts(normalize=True))) #Conta o numero de ocorrencias de cada classe
print("Treino: " + str(y_train.value_counts(normalize=True)))
print("Teste: " + str(y_test.value_counts(normalize=True)))

plt.figure(figsize=(10, 5))
df_origin_dist = y.value_counts(normalize=True).reset_index()
df_origin_dist['Dataset'] = 'Original'
df_train_dist = y_train.value_counts(normalize=True).reset_index()
df_train_dist['Dataset'] = 'Treino'
df_test_dist = y_test.value_counts(normalize=True).reset_index()
df_test_dist['Dataset'] = 'Teste'

df_dist = pd.concat([df_origin_dist, df_train_dist, df_test_dist])
sns.barplot(data=df_dist, x='Result', y='proportion', hue='Dataset')
plt.title('Distribuição das Classes: Original vs Treino vs Teste')
plt.ylabel('Proporção')
plt.show()


#%%
# ============================EXPLORE=============================
# ['Análise Descritiva', 'Análise Bivariada', 'Identificação de Missing']


X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
#%%

# VALIDANDO GAP DE DADOS (Identificação de Missing)
X_train.isna().sum().sort_values(ascending=False)
#%%

# EXPLORANDO O COMPORTAMENTO DAS FEATURES EM CADA RESPOSTA (Análise Descritiva)
df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(['mean', 'median']).T
sumario
# CASO TIVESSEMOS UM CONTROLE, PODERIAMOS ANALISAR A DIFERENÇA RELATIVA DE EXPRESSÃO PARA CADA CASO DA SEGUINTE FORMA
# sumario['diff_rlt_BRCA'] = sumario['BRCA'] - sumario['Controle']
# sumario.sort_values(by=['diff_rlt_BRCA'], ascending=False)
# ASSUMINDO 'LUAD (Adenocarcinoma pulmonar)' COMO NOSSO CONTROLE PARA FINS DE TESTE TEMOS:
sumario['diff_rlt_BRCA'] = sumario['BRCA'] / sumario['LUAD']
sumario
#%%



# CRIANDO UMA DECISIONTREE PARA SABER OS GENES MAIS IMPORTANTES
arvore = tree.DecisionTreeClassifier(random_state=42, max_depth=3) #'max_depth=3' apenas devido a limitação gráfica para plotagem da arvore
arvore.fit(X_train, y_train)
tree.plot_tree(arvore, feature_names=X_train.columns, filled=True, class_names=arvore.classes_)
plt.figure(dpi=700, figsize=[4,4])

#%%
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)
feature_importances = (pd.Series(arvore.feature_importances_, 
                                 index=X_train.columns)
                                 .sort_values(ascending=False)
                                 .reset_index())
#%%
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances.head(10), x=0, y='index', palette='viridis')
plt.title('Top 10 Genes por Importância')
plt.xlabel('Importância Relativa')
plt.ylabel('Gene')
plt.show()

#%%
feature_importances['acum'] = feature_importances[0].cumsum()
feature_importances
#%%
best_features = feature_importances[feature_importances['acum'] < 0.96]['index'].to_list() #Filtra os valores que somam 95% da importância
best_features


# ============================MODIFY=============================
# ['Padronização', 'Imputação de Missing', 'Binning', 'Combinação']



#%%
# TRANFORMA A VARIAVEL DO TIPO QUALITATIVA NOMINAL PARA QUANTITATIVA DISCRETA
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
mapping

#%%
tree_discretisation = discretisation.DecisionTreeDiscretiser(variables=best_features, regression=False)
tree_discretisation.fit(X_train, y_train_encoded)

#%%
# PADRONIZA AS FEATURES
X_train_trasformed = tree_discretisation.transform(X_train)
X_test_trasformed = tree_discretisation.transform(X_test)
X_test_trasformed



# ============================MODEL=============================
# ['Modelos Estatísticos', 'Modelo de Árvore', 'Modelo de Vetor de Suporte', 'Redes Neurais']


#%%
reg = linear_model.LogisticRegression(penalty=None, random_state=42)
reg.fit(X_train_trasformed, y_train_encoded)



# ============================ASSESS=============================
# ['Métricas de Ajuste', 'Decisão', 'Comparação', 'Serialização']



#%%
y_train_predict = reg.predict(X_train_trasformed)
y_train_proba = reg.predict_proba(X_train_trasformed)
acc_train = metrics.accuracy_score(y_train_encoded, y_train_predict)
auc_train = metrics.roc_auc_score(y_train_encoded, y_train_proba, multi_class='ovr')
print("Acuracia Treino: " + str(acc_train))
print("AUC Treino: " + str(auc_train))
#%%
y_test_predict = reg.predict(X_test_trasformed)
y_test_proba = reg.predict_proba(X_test_trasformed)
acc_test = metrics.accuracy_score(y_test_encoded, y_test_predict)
auc_test = metrics.roc_auc_score(y_test_encoded, y_test_proba, multi_class='ovr')
print("Acuracia Treino: " + str(acc_test))
print("AUC Treino: " + str(auc_test))
#%%

plt.figure(figsize=(8, 6))
for i in range(len(le.classes_)):
    fpr, tpr, _ = roc_curve(y_test_encoded == i, y_test_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--') # Linha de referência diagonal
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Desempenho por Tipo de Câncer')
plt.legend(loc='lower right')
plt.show()







