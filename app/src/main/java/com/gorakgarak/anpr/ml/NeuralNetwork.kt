package com.gorakgarak.anpr.ml

import android.content.Context
import com.gorakgarak.anpr.R
import com.gorakgarak.anpr.parser.GorakgarakXMLParser
import org.opencv.core.Core
import org.opencv.core.CvType.CV_32FC1
import org.opencv.core.CvType.CV_32SC1
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.ml.ANN_MLP
import org.opencv.ml.Ml.ROW_SAMPLE
import java.io.FileInputStream

/**
 * Created by kohry on 2017-10-15.
 */
object NeuralNetwork {

    val CHAR_COUNT = 30
    val LAYER_COUNT = 10

    val strCharacters = arrayOf('0','1','2','3','4','5','6','7','8','9','B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z')

    val ann: ANN_MLP = ANN_MLP.create()

    private fun readXML(context: Context): Pair<Mat, Mat> {
//        val inputStream = context.assets.open(fileName)
//        val fs = opencv_core.FileStorage()
//        fs.open(fileName,opencv_core.FileStorage.READ)
//
//        val train = Mat(fs["TrainingDataF15"].mat().address())
//        val classes = Mat(fs["classes"].mat().address())
        return GorakgarakXMLParser.parse(context.resources.openRawResource(R.raw.ann),"TrainingDataF15")

    }

    fun train(context: Context) {

        if (ann.isTrained) return

        val data = readXML(context)

        val trainData = data.first
        val classes = data.second

        val layerSizes = Mat(1,3,CV_32SC1)
        val r = trainData.rows()
        val c = trainData.cols()

        layerSizes.put(0, 0, intArrayOf(trainData.cols()))
        layerSizes.put(0, 1, intArrayOf(LAYER_COUNT))
        layerSizes.put(0, 2, intArrayOf(CHAR_COUNT))

        ann.layerSizes = layerSizes
        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM)

        val trainClasses = Mat()
        trainClasses.create(trainData.rows(), CHAR_COUNT, CV_32FC1)
        (0 until trainClasses.rows()).forEach { row ->
            (0 until trainClasses.cols()).forEach { col ->
                if (col == classes.get(row, 0).get(0).toInt()) trainClasses.put(row, col, floatArrayOf(1f))
                else trainClasses.put(col, row, floatArrayOf(0f))
            }
        }
        //this part has changed from opencv 2 -> 3. ann class does not need weights anymore.
//        val weights = Mat(1, trainData.rows(), CV_32FC1, Scalar.all(1.0))
        ann.train(trainData, ROW_SAMPLE, trainClasses)

    }

    fun classify(f: Mat): Double {
        val result = -1
        val output = Mat(1, CHAR_COUNT, CV_32FC1)
        ann.predict(f)
        val minMaxLoc = Core.minMaxLoc(output)

        return minMaxLoc.maxLoc.x

    }


}