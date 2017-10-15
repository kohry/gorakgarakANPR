package com.gorakgarak.anpr.ml

import org.opencv.core.Core
import org.opencv.core.CvType.CV_32FC1
import org.opencv.core.CvType.CV_32SC1
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.ml.ANN_MLP

/**
 * Created by kohry on 2017-10-15.
 */
object ANN {

    val strCharacters = arrayOf('0','1','2','3','4','5','6','7','8','9','B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z')
    val numCharacters = 30
    val ann = ANN_MLP.create()

    fun train(trainData: Mat, classes: Mat, nlayers: Int) {
        val layerSizes = Mat(1,3,CV_32SC1)
        val r = trainData.rows()
        val c = trainData.cols()

        (0 until r).forEach { layerSizes.put(0, it, trainData.get(0, it).map { it.toInt() }.toIntArray()) }
        (0 until r).forEach { layerSizes.put(1, it, nlayers) }
        (0 until r).forEach { layerSizes.put(2, it, numCharacters) }

        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM)

        val trainClasses = Mat()
        trainClasses.create(trainClasses.rows(), numCharacters, CV_32FC1)
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
        val output = Mat(1, numCharacters, CV_32FC1)
        ann.predict(f)
        val minMaxLoc = Core.minMaxLoc(output)

        return minMaxLoc.maxLoc.x

    }


}