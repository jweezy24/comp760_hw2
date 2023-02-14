# Decision Tree Homework

Jack West's repo for Homework 2 in 761. If there are any complications, email me at `jwwest@wisc.edu`.

## Dependencies

To install dependencies, run the command below.
`` pip install -r requirements.txt``

Be aware that I use `python` as the python command.
**This code requires Python 3** for some systems, python will be located at the `python3` command.

## Run the code
To run the code, you must exrecute `python ./main.py` from the root directory. Running the code will not give all question solutions. 

To change the dataset, you must change the condtion on line **194** to select the database you wish to evaluate.

When working with `Dbig.txt`, you mus also uncomment the `big_dataset_logic` and comment out the `normal_dataset_logic`.
To observe the sklearn tree, the process is the same except uncommenting the `sklearn_tree_analysis`.
A note about the sklearn question, the datasets are built with uniform random sampling on each script run, thus the results will slightly differ per run.

When validating `bad_dataset.txt` you must change line **110** from `t = DecisionTree(dataset[key],name=key,force_choice=False,choice_ind=2)` to `t = DecisionTree(dataset[key],name=key,force_choice=True,choice_ind=2)`.

When testing question 4, you must execute the command `python ./lagrange_interpolation.py`