import math
import numpy as np
from perceptron import perceptron
from confus import confus
from linmach import linmach
import pandas as pd
from alive_progress import alive_bar

def compute_and_save_params(dataFile, alphas, bs):
  parametersDf = pd.DataFrame(columns=['dataFile', 'a', 'b', 'E', 'k', 'Ete', 'Ete (%)', 'Ite (%)'])
      # a is learning rate
      # b is margin
      # E is misclassified samples in the training set with Perceptron (Etr)
      # k is Iterations Perceptron (epochs)
      # Ete is number of errors in the training set
      # Ete (%) is the estimated error in percent
      # Ite (%) is the confidente interval in percent
  data = np.loadtxt(f'data/{dataFile}')
  N,L = data.shape
  D=L-1
  labs = np.unique(data[:,L-1])
  C = labs.size
  np.random.seed(23)
  perm = np.random.permutation(N)
  data = data[perm]
  NTr = int(round(.7*N))
  train = data[:NTr,:]
  M = N-NTr
  test = data[NTr:,:]
  
  print(f"\nComputing results for the {dataFile} dataset")
  with alive_bar(len(alphas)*len(bs)) as bar:
    for a in alphas:
      for b in bs:
        w,E,k=perceptron(train,b,a)
        rl=np.zeros((M,1))
        for n in range(M):
          rl[n] = labs[linmach(w,np.concatenate(([1], test[n,:D])))]
        nerr,m = confus(test[:,L-1].reshape(M,1), rl)

        Ete = nerr
        per = Ete/M
        r = 1.96*math.sqrt(per * (1-per) /M)
        EtePercent = round(Ete/M*100, 1)
        ItePercent = [round((per-r)*100, 1), round((per+r)*100, 1)]

        parametersDf.loc[len(parametersDf)] = [dataFile, a, b, E, k, Ete, EtePercent, ItePercent]
        bar()

  # save df to results/dataFile.csv
  parametersDf.to_csv(f'results/{dataFile}.csv', index=False)
  print(f"Results successfully saved to the file 'results/{dataFile}.csv'.")
  print_result_table(dataFile)


def compute_and_save_all_results(datasetNames, alphas, bs):
  for datasetName in datasetNames:
    compute_and_save_params(datasetName, alphas, bs)

def print_result_table(datasetName):
  results_df = pd.read_csv(f'results/{datasetName}.csv')
  print(f"\nResult table with the all of the results of dataset {datasetName}:")
  print(results_df)

def print_all_result_tables(datasetNames):
  for datasetName in datasetNames:
    print_result_table(datasetName)
    print('\n')

def save_best_results_to_summary_table(datasetNames):
  # get the best parameters for each dataset and put the in a results df
  summary_df = pd.DataFrame(columns=['dataFile', 'a', 'b', 'E', 'k', 'Ete', 'Ete (%)', 'Ite (%)'])
  for datasetName in datasetNames:
    parametersDf = pd.read_csv(f'results/{datasetName}.csv')
    best_row = parametersDf.loc[parametersDf['Ete (%)'].idxmin()]
    summary_df.loc[len(summary_df)] = best_row
  # save summary_df to results/summary.csv
  summary_df.to_csv(f'results/summary.csv', index=False)

def print_summary_table_with_all_columns():
  result_df = pd.read_csv('results/summary.csv')
  print("\nSummary table with the best approximate results (lowest Ete (%)):")
  print(result_df)

def print_summary_table_with_selected_columns():
  summary_df = pd.read_csv('results/summary.csv')
  print("\nSummary table with the best approximate results (lowest Ete (%)):")
  print(summary_df[['dataFile', 'Ete (%)', 'Ite (%)']])

def main():
  compute_and_save_all_results(datasetNames, alphas, bs)
  save_best_results_to_summary_table(datasetNames)
  print_summary_table_with_all_columns()

# Global variables
alphas = np.fromstring('.1 1 10 100 1000 10000', sep=' ') # '.1 1 10 100 1000 10000'
bs = np.fromstring('.1 1 10 100 1000 10000 100000', sep=' ') # '.1 1 10 100 1000 10000 100000'
datasetNames = ["OCR_14x14", "expressions.gz", "gauss2D.gz", "gender.gz", "videos.gz"]

# Main function
main()

### Explanation ### 
# Execute main to do everything described under.
################################################
# Execute compute_and_save_all_results(datasetNames, alphas, bs) to compute the results tables for all datasets in the datasetNames list using the given alphas and bs.
#   this takes some time to finish. The progressbar will show approx. time left. After each dataset is computed and saved, it will be printed out.
# Execute save_best_results_to_summary_table() to save all the best results (lowest Ete (%)) for each dataset in a summary table.
# Execute print_summary_table_with_all_columns() to print the summary table with all columns, 
#   or execute print_summary_table_with_selected_columns() to pring the summary table with selected columns,

### Additional info ###
# All saved tables can be viewed in the results folder as csv files.