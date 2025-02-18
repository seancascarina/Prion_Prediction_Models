
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
import pickle
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'


def main():

    # GET PRION AND SEQUENCE DATA=============================================#
    seqs_df = get_seqs()
    df = pd.read_csv('Alberti2009_PrLDscores.tsv', sep='\t')
    df = df[df['Sup35 Assay'].notna()]
    sup35_target = np.array( [1 if x>0 else 0 for x in df['Sup35 Assay']] )

    orf_names = list(df['Orf Name'])
    for orf in orf_names:
        if orf not in seqs_df:
            print('Missing orf:', orf)
    orf_names = [x for x in orf_names if x in seqs_df]
    seqs = [seqs_df[orf] for orf in orf_names]
    #=========================================================================#

    # EXTRACT AMINO ACID COMPOSITION FEATURES FROM SEQUENCES==================#
    comp_matrix = [[seq.count(aa) / len(seq) *100 for aa in amino_acids] for seq in seqs]
    df = pd.DataFrame(comp_matrix, columns=list(amino_acids))
    #=========================================================================#

    # THE lbfgs SOLVER WAS USED DUE TO THIS NOTE IN THE SCIKIT-LEARN DOCUMENTATION:
    # Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands 
    # of training samples or more) in terms of both training time and validation score. For small datasets, 
    # however, ‘lbfgs’ can converge faster and perform better
    params_grid = {'hidden_layer_sizes':[],
                # 'alpha':np.logspace(0.01, 100, num=200),  # INITIAL RANGE GUESS YIELDED POOR PREDICTORS THAT PREDICTED ALL SEQUENCES TO BE NON-PRION
                'alpha':np.logspace(-4, -0.5, num=200),  # YIELDS FEWER POOR PREDICTORS.
                'solver':['lbfgs'],
                'activation':['logistic']}
                
    # MAKE COMBINATIONS OF LAYER NUMBERS AND SIZES=========================#
    n_layers = [x for x in range(1,3)]
    n_nodes = [x for x in range(2,41)]
    for n_layer in n_layers:
        layer_list = []
        for n_node in n_nodes:
            layer_list.append( n_node )
            params_grid['hidden_layer_sizes'].append((n_layer, n_node))
    #======================================================================#
            
    scoring = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']
        
    # RUN RandomizedSearchCV TO FIND BEST MODEL, THEN OUTPUT RESULTS=======#
    classifier = MLPClassifier(random_state=11, max_iter=10000)
    clf_rs = RandomizedSearchCV(classifier, params_grid, n_iter=1000, refit='f1', random_state=0, scoring=scoring)
    fit_rs = clf_rs.fit(df, sup35_target)
    output_results_model(fit_rs)
    #======================================================================#
    
    
def output_results_model(fit_rs):
    """Output complete RandomizedSearchCV cross validation model results to a csv file,
    as well as the final best model (according to F1 score) to a binary pickle file.
    """
    
    cv_results = fit_rs.cv_results_
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv('NeuralNetwork_PrionPrediction_cv_results.csv', index=False)
    
    pf = open('NeuralNetwork_PrionPrediction_BestModel.pickle', 'wb')
    best_model = fit_rs.best_estimator_
    pickle.dump(best_model, pf)
    pf.close()
    
    
def get_seqs():
    """Get sequences for PrLDs.
    Returns:
        df = dictionary of orf_name:sequence key value pairs.
    """
    
    h = open('Alberti_full_PrLDs_ALLseqs.csv')
    header = h.readline()
    
    df = {}
    for line in h:
        gene, orf_name, common, seq, *remainder = line.rstrip().split(',')
        if orf_name in df:
            print('DUPLICATE:', orf_name)
            exit()
        df[orf_name] = seq
        
    h.close()
    
    return df


if __name__ == '__main__':
    main()