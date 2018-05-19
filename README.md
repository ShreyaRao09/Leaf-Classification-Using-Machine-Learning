# Leaf-Classification-Using-Machine-Learning
A Machine Learning Project that uses strong and powerful learners to predict plant species.

There are about one million species of plants in the world. Classification of leaf species has been problematic for many years now, especially when the same leaf species are counted and identified more than once, creating duplicates entries. Automating plant recognition helps biologists in tracking and preserving species population, research on plant based medicines and crop and food supply management. The leaf species are considered because of their volume, prevalence and unique characteristics that help differentiate leaves of one plant from another.

The dataset consists of 1584 images of leaf specimen. These images have been converted to white colored leaves of binary format against a black background. Each image has 3 sets of features which are shape contiguous descriptor for shape feature, interior texture histogram for texture feature and fine-scale margin histogram for margin feature. Each of these features consists of a 64- attribute vector per leaf sample

The approach has been explained in the project report.


Instructions to run the code
1. In line 8, replace the path of the file with the full path of "train.csv"
2. In line 9, replace the path of the file with the full path of "test.csv"
3. In line 158, replace the path of the “file” parameter with any path (it should include a .csv file name) of your choice. This function will create a .csv file of the same name
4. Run

Note:
1. Each classifier takes a long time to run
2. At the end of the classifiers, we have executed the "test.csv" file that contains unlabled test data that was provided by the competition.

