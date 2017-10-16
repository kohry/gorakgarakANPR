package com.gorakgarak.anpr.ml

import android.content.Context
import android.util.Log
import com.gorakgarak.anpr.MainActivity
import org.opencv.core.Mat
import org.opencv.core.TermCriteria
import org.opencv.ml.Ml.ROW_SAMPLE
import org.opencv.ml.SVM
import java.io.File

/**
 * Created by kohry on 2017-10-16.
 */
object SupportVector {

    private const val TAG = "SVM Object"
    private val svm: SVM? = null

    fun getSvmClassifier(): SVM? = svm

    //Sadly, Android opencv sdk does not support FileStorage.
    //This is one method to read XML file from assets.
    //Pretty sucks.
    private fun readXML(context:Context, fileName: String): Pair<Mat, Mat> {
        val inputStream = context.assets.open(fileName)
        val train = Mat()
        val classes = Mat()

//        train.put

        return Pair(train, classes)

    }

    //Train data when application first turn on.
    //Actually, this part should be outside of the Android system.
    //Don't you think?? KKUL KKUL
    fun train(context: Context) {

        val data = readXML(context, "OCR.xml")

        Log.d(TAG, "Set initial SVM Params")
        val dataMat = data.first
        val classes = data.second

        var svm = SVM.create()
        svm.type = SVM.C_SVC
        svm.degree = 0.0
        svm.gamma = 1.0
        svm.coef0 = 0.0
        svm.c = 1.0
        svm.nu = 0.0
        svm.p = 0.0
        svm.termCriteria = TermCriteria(TermCriteria.MAX_ITER, 1000, 0.01)
        svm.setKernel(SVM.LINEAR)

        svm.train(dataMat ,ROW_SAMPLE, classes)
        svm.isTrained

    }

}