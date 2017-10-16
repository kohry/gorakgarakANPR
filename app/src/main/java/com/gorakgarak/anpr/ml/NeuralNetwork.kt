package com.gorakgarak.anpr.ml

import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType.CV_32FC1
import org.opencv.core.CvType.CV_32SC1
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.ml.ANN_MLP

/**
 * Created by kohry on 2017-10-15.
 */
object NeuralNetwork {

    val CHAR_COUNT = 30
    val LAYER_COUNT = 10

    val strCharacters = arrayOf('0','1','2','3','4','5','6','7','8','9','B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z')

    val ann: ANN_MLP = ANN_MLP.create()

    private fun readXML(context: Context, fileName: String): Pair<Mat, Mat> {
        val inputStream = context.assets.open(fileName)
        val train = Mat()
        val classes = Mat()

//        train.put

        return Pair(train, classes)

    }

    fun train(context: Context) {

        val data = readXML(context, "OCR.xml")

        val trainData = data.first
        val classes = data.second

        val layerSizes = Mat(1,3,CV_32SC1)
        val r = trainData.rows()
        val c = trainData.cols()

        (0 until r).forEach { layerSizes.put(0, it, trainData.get(0, it).map { it.toInt() }.toIntArray()) }
//        (0 until r).forEach { layerSizes.put(1, it, nlayers) }
//        (0 until r).forEach { layerSizes.put(2, it, CHAR_COUNT) }

        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM)

        val trainClasses = Mat()
        trainClasses.create(trainClasses.rows(), CHAR_COUNT, CV_32FC1)
        (0 until trainClasses.rows()).forEach { row ->
            (0 until trainClasses.cols()).forEach { col ->
                //:TODO 이거 도대체 어떠케 하는거야? at method 잘알필요있다.
            }
        }

        val weights = Mat(1, trainData.rows(), CV_32FC1, Scalar.all(1.0))
        ann.train(trainData, 0, weights)  //:TODO 여기서 가운데 인자 뭘로 줘야하는가? trainclasses는 쓰이지가 않는다.

    }

    fun classify(f: Mat): Double {
        val result = -1
        val output = Mat(1, CHAR_COUNT, CV_32FC1)
        ann.predict(f)
        val minMaxLoc = Core.minMaxLoc(output)

        return minMaxLoc.maxLoc.x

    }


}