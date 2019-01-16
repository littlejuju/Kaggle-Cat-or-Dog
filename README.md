## README
### GoogLeNet Feature Extraction
* *traincat.mat* Features extracted based on GoogLenet through Matlab2017b. Matrix of cat pictures, including 1981 cat pictures, each of which is extracted into a 1024 x 1 vector. 
* *traindog.mat* The same as above. Matrix of dog pictures，including 1930 dog pictures, each of which is extracted into a 1024 x 1 vector. 
* *test25000.mat* Extracted features of 25000 test cat and dog pictures using trained GoogLenet model, which is a 1024 x 25000 matrix. former 12500 of them are cats, latter 25000 are dogs.

### Classifiers
* catordogadboost.py: AdaBoost
* catordoggbdt.py: GBDT
* catordog.py: XGBoost

Cross validation is in commented *cv part*.
