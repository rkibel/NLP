Ron Kibel- PERM 5166087

Note: the code closely follows the tutorial at https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html, but the feature template is changed

Make sure this folder contains: 3 txt files (test, train, and validation) and 1 csv file (test_noans.csv)

The csv file is for formatting purposes: it is easier to convert this csv to a pandas DF than a txt file

To run the Spanish NER, navigate to this spanish_ner folder locally and run the command "./run.sh test.txt test_ans.csv"

The first argument is the name of the read-in test file (make sure this is a txt)

The second argument is the name of the output file (make sure this a csv)