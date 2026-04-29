#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import tree
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import metrics
from sklearn import pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from feature_engine import discretisation, encoding
import mlflow

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

plt.figure(dpi=700, figsize=[6,3])
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
plt.figure(dpi=700, figsize=[6,3])
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
# No caso em questão, isso se aplica apenas a variável target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
mapping

#%% **************ESSE BLOCO É APENAS UM EXEMPLO*******************
# PADRONIZA AS FEATURES (DISCRETIZAÇÃO)
# No caso em questão não mudara nada pois as features já são numericas, mas no caso de features nominais isso é util
tree_discretisation = discretisation.DecisionTreeDiscretiser(variables=best_features, regression=False)
tree_discretisation.fit(X_train, y_train_encoded)

X_train_trasformed = tree_discretisation.transform(X_train)
X_test_trasformed = tree_discretisation.transform(X_test)
X_test_trasformed

# ONEHOTENCODING
# Nesse ponto as features ordinais foram transformadas em numéricas, 
# mas com valores como [1, 2, 3...] o que pode sugerir que são sequenciais, e não são.
# Para corrigir isso, vamos fazer onehotEncoding para dividir esses valores em colunas de 0 e 1
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)
onehot.fit(X_train_trasformed, y_train_encoded)
X_train_trasformed = onehot.transform(X_train_trasformed)
X_train_trasformed
# Isso vai almentar muito o número de colunas, mas após isso você pode ranquear as colunas mais importantes e dropar as menos.



# ============================MODEL=============================
# ['Modelos Estatísticos', 'Modelo de Árvore', 'Modelo de Vetor de Suporte', 'Redes Neurais']

# EXEMPLOS DE MODELOS
model = linear_model.LogisticRegression(penalty=None, random_state=42)
model = naive_bayes.BernoulliNB()
model = ensemble.RandomForestClassifier(random_state=42, 
                                        min_samples_leaf=28,
                                        n_jobs=-1, #Número de CPUs a serem usadas ao mesmo tempo (-1 = todas as disponiveis).
                                        n_estimators=500) #Número de arvores a serem criadas.
model - tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=28)
model = ensemble.AdaBoostClassifier(random_state=42,
                                    n_estimators=500,
                                    learning_rate=0.01) #Grau de penalidade aplicados as arvores que o modelo errou durante seu treinamento.

# EXEMPLO DE PIPELINE
model_pipeline = pipeline.Pipeline(
    steps=[
        {'Discretization': tree_discretisation},
        {'Onehot': onehot},
        {'Model': model},
    ]
)

model_pipeline.fit(X_train, y_train)
# Nesse caso, tudo que se segue para baixo seria simplificado, 
# o model seria substituido pelo model_pipeline 
# e o X_test_trasformed e y_train_encoded substituidos por X_train e y_train


#%% *****************EXECUTE A PARTIR DESSE BLOCO***************************
# ESSE SERA O MODELO QUE IREMOS ADOTAR COMO EXEMPLO
model = linear_model.LogisticRegression(penalty=None, random_state=42)
model.fit(X_train_trasformed, y_train_encoded)

# ============================ASSESS=============================
# ['Métricas de Ajuste', 'Decisão', 'Comparação', 'Serialização']


#%%
y_train_predict = model.predict(X_train_trasformed)
y_train_proba = model.predict_proba(X_train_trasformed)
acc_train = metrics.accuracy_score(y_train_encoded, y_train_predict)
auc_train = metrics.roc_auc_score(y_train_encoded, y_train_proba, multi_class='ovr')
print("Acuracia Treino: " + str(acc_train))
print("AUC Treino: " + str(auc_train))
#%%
y_test_predict = model.predict(X_test_trasformed)
y_test_proba = model.predict_proba(X_test_trasformed)
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

plt.plot([0, 1], [0, 1], 'k--') 
plt.title('Curva ROC - Desempenho por Tipo de Câncer')
plt.xlabel("1 - Especificidade") # Taxa de Falsos Positivos
plt.ylabel("Sensibilidade") # Taxa de Verdadeiros Positivos
plt.legend(loc='lower right')
plt.show()







#%% **************ESSE BLOCO É APENAS UM EXEMPLO*******************
# ============================MLFlow=============================
# O QUE SE SEGUE AQUI É APENAS UM EXEMPLO DE IMPLEMENTAÇÃO E NÃO ESTÁ RELACIONADO AO EXEMPLO ACIMA
# O MLFlow é uma aplicação que ajuda a gerencias os varios modelos que forem criados/treinados.
# Ele faz isso atravez de uma interfacie web, onde é possivel comparar a eficácia dos modelos facilmente.

# P/ SUBIR A APLICAÇÃO => mlflow server
# Na aplicação é necessário criar um novo experimento e pegar o ID do experimento
# No código, para logar o modelo no mlflow, adicione as linhas a seguir antes do treinamento do modelo

mlflow.set_tracking_uri("porta_onde_o_mlflow_ta_rodando")
mlflow.set_experiment(experiment_name="nome_do_experimento")

with mlflow.start_run(run_name=model.__str__()):
    
    mlflow.sklearn.autolog()

    model_pipeline.fit(X_train, y_train)

    # ...
    # TODA A PARTE DE ASSESS, TESTE DE ACURACIA E AUC VEM AQUI
    # ...

    mlflow.log_metrics({
    "acc_train":acc_train,
    "auc_train":auc_train,
    "acc_test":acc_test,
    "auc_test":auc_test,
    })

