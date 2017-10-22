# Simple Alertanative Code for OpenCV FileStorage.

You are reading masteringOpenCV book, and frustrated with the fact that [OpenCV3 Android SDK] has no FileStorage class in it? This simple Kotlin code modified from [Android tutorial](https://developer.android.com/training/basics/network-ops/xml.html) can help you.

Call parse() method in this code, and you will get a nice decent Pair<Mat, Mat> object from it. (Training Data Mat & Classes Mat).


## Example
GorakgarakXMLParser.parse(context.resources.openRawResource(R.raw.svm),"TrainingData")



## Others..
Sample SVM training data in xml format is in the same directory. This is a sub-project for [Gorakgarak Automated Number Plate Recognition](https://github.com/kohry/gorakgarakANPR) which is in progress.



## Version
- Kotlin 1.1.51
- OpenCV 3