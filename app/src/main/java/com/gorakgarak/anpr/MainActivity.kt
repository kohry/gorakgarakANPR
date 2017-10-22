package com.gorakgarak.anpr

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.support.v4.app.ActivityCompat
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Toast
import com.gorakgarak.anpr.ml.NeuralNetwork
import com.gorakgarak.anpr.ml.NeuralNetwork.CHAR_COUNT
import com.gorakgarak.anpr.ml.NeuralNetwork.strCharacters
import com.gorakgarak.anpr.ml.SupportVector
import com.gorakgarak.anpr.model.CharSegment
import com.gorakgarak.anpr.model.Plate
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.*
import org.opencv.core.*
import org.opencv.core.Core.*
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.*
import java.util.*
import org.opencv.core.CvType
import org.opencv.core.CvType.*
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint
import org.opencv.core.RotatedRect


class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private var _cameraBridgeViewBase: CameraBridgeViewBase? = null

    private val _baseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    // Load ndk built module, as specified in moduleName in build.gradle
                    // after opencv initialization
                    System.loadLibrary("native-lib")
                    _cameraBridgeViewBase!!.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        // Permissions for Android 6+
        ActivityCompat.requestPermissions(this@MainActivity,
                arrayOf(Manifest.permission.CAMERA),
                1)

        _cameraBridgeViewBase = findViewById(R.id.main_surface) as CameraBridgeViewBase
        _cameraBridgeViewBase!!.visibility = SurfaceView.VISIBLE
        _cameraBridgeViewBase!!.setCvCameraViewListener(this)


    }


    public override fun onPause() {
        super.onPause()
        disableCamera()
    }

    public override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, _baseLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            _baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }

        //When first created, train SVM Data. Don't forget to show progress wheel.
//        SupportVector.train(this@MainActivity)

        //train OCR.xml by Artificial Neural Network
//        NeuralNetwork.train(this@MainActivity)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        when (requestCode) {
            1 -> {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.size > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                } else {
                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                    Toast.makeText(this@MainActivity, "Permission denied to read your External storage", Toast.LENGTH_SHORT).show()
                }
                return
            }
        }// other 'case' lines to check for other
        // permissions this app might request
    }

    public override fun onDestroy() {
        super.onDestroy()
        disableCamera()
    }

    fun disableCamera() {
        if (_cameraBridgeViewBase != null)
            _cameraBridgeViewBase!!.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {}

    override fun onCameraViewStopped() {}

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {

        //        salt(matGray.getNativeObjAddr(), 2000);

        Log.d(TAG, "1-1) GrayScale & blur for noise removal")
        val matGray = inputFrame.gray()
        Imgproc.blur(matGray, matGray, Size(5.0, 5.0))

        Log.d(TAG, "1-2) Sobel")
        val matSobel: Mat = Mat()
        Imgproc.Sobel(matGray, matSobel, CvType.CV_8U, 1, 0, 3, 1.0, 0.0)

        Log.d(TAG, "1-3) Threshold")
        val matThreshold: Mat = Mat()
        Imgproc.threshold(matSobel, matThreshold, 0.0, 255.0, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY)

        Log.d(TAG, "1-4) Threshold")
        val element: Mat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(17.0, 3.0))
        Imgproc.morphologyEx(matThreshold, matThreshold, Imgproc.MORPH_CLOSE, element)

        Log.d(TAG, "1-5) Find contour of possible plates")
        val contourList: List<MatOfPoint> = mutableListOf()
        val hierarchy = Mat()
        Imgproc.findContours(matThreshold, contourList, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)

        Log.d(TAG, "1-6) Get rectangle from the contours")
        val rectList: MutableList<RotatedRect> = mutableListOf()

        //Convert MapOfPoint to MatOfPoint2f
        contourList.forEach {
            val p = MatOfPoint2f()
            it.convertTo(p, CvType.CV_32F)
            val img = Imgproc.minAreaRect(p)
            if (verifySizes(img)) rectList.add(Imgproc.minAreaRect(p))
        }

        //draw contour on original colored image to fetch white number plate.
        val input = inputFrame.rgba()
        cvtColor(input, input, COLOR_RGBA2RGB)
        val result = Mat()
        input.copyTo(result)
        //So many contours detected
//        drawContours(result, contourList, -1, Scalar(200.0, 0.0, 0.0), 1) // more than 100~

        val logoMat = Utils.loadResource(this, R.mipmap.ic_launcher)
        cvtColor(logoMat, logoMat, COLOR_RGBA2RGB)

        rectList.forEach { rect ->
            //temp rectangle to findout the rectangle candidate. mostly 3~100
            rectangle(result, rect.boundingRect().tl(), rect.boundingRect().br(), Scalar(0.0, 200.0, 0.0))
            putText(result, "Xeed Lab Detected!", rect.boundingRect().tl(), FONT_HERSHEY_COMPLEX, 0.8, Scalar(200.0, 0.0, 0.0), 2)
        }

        Log.d(TAG, "1-7) Floodfill algorithm from more clear contour box, get plates candidates")
        val plateCandidates = getPlateCandidatesFromImage(input, result, rectList)

        Log.d(TAG, "2-2) Using trained svmClassifier, let's predict number plates")
        val plates = mutableListOf<Plate>()
        plateCandidates.forEach { candidate ->
            val p = candidate.img.reshape(1, 1)
            p.convertTo(p, CV_32FC1)
            val response = SupportVector.getSvmClassifier()?.predict(p) ?: 0
            if (response == 1f) plates.add(candidate)
        }
//
        Log.d(TAG, "${plates.size} plates has been detected")

        Log.d(TAG, "3-1) with predicted car plates and trained ANN Classifier, get Strings")
        plates.forEach { plate ->
            plate.str = ""
            val charSegments = getCharSegmentFromOcr(plate.img) //remeber this is not characeter but image segments
            charSegments.forEach { charCandidate ->
                val ch = preprocessChar(charCandidate.image)
                val f = getFeatures(ch, 15.0)
                val charResult = classifyWithANNModel(f)
                val str = strCharacters[charResult.toInt()]
                plate.str = plate.str + str
            }
            text_numberplate.text = plate.str
        }

        return result
    }

    private fun classifyWithANNModel(f: Mat): Double {
        var result = -1
        val output = Mat(1, CHAR_COUNT, CV_32FC1)
        NeuralNetwork.ann.predict(f)
        val loc = minMaxLoc(output)
        return loc.maxLoc.x
    }

    private fun getCharSegmentFromOcr(plate: Mat): List<CharSegment> {

        val output: MutableList<CharSegment> = mutableListOf()

        val thresholdMat = Mat()
        threshold(plate, thresholdMat, 60.0, 255.0, THRESH_BINARY_INV)

        val contouredMat = Mat()
        thresholdMat.copyTo(contouredMat)

        //Find contours of possibles characters
        val contourList: List<MatOfPoint> = mutableListOf()
        val hierarchy = Mat()
        Imgproc.findContours(contouredMat, contourList, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)

        //Find contours of possibles characters
        val result = Mat()
        thresholdMat.copyTo(result)
        cvtColor(result, result, COLOR_GRAY2RGB)
        drawContours(result, contourList, -1, Scalar(255.0, 0.0, 0.0), 1)

        //Remove patch that are no inside limits of aspect ratio and area.
        contourList.forEach { contour ->

            val mr: Rect = boundingRect(contour)
            rectangle(result, mr.tl(), mr.br(), Scalar(255.0, 0.0, 0.0))

            var auxRoi = Mat(thresholdMat, mr)
            if (verifySizeForChar(auxRoi)) {
                auxRoi = preprocessChar(auxRoi)
                output.add(CharSegment(auxRoi, mr))
                rectangle(result, mr.tl(), mr.br(), Scalar(0.0, 125.0, 255.0))
            }
        }
        return output
    }

    fun preprocessChar(input: Mat): Mat {
        val h = input.rows()
        val w = input.cols()
        val transformMat = Mat.eye(2, 3, CV_32F)
        val m = Math.max(w, h)
        transformMat.put(0, 2, (m / 2 - w / 2).toDouble())
        transformMat.put(1, 2, (m / 2 - w / 2).toDouble())

        val warpImage = Mat(m, m, input.type())
        warpAffine(input, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0.0))
        val output = Mat(0)
        val charSize = 20.0
        resize(warpImage, output, Size(charSize, charSize))

        return output
    }

    //TODO: t 가 뭔지 살펴봐야함.
    fun getProjectedHistogram(img: Mat, t: Int): Mat {
        val size = if (t == 1) img.rows() else img.cols()
        val mhist = Mat.zeros(1, size, CV_32F)

        (0 until size).forEach {
            val data = if (t == 1) img.row(it) else img.col(it)
            mhist.put(0, it, intArrayOf(countNonZero(data)))
        }

        val loc = minMaxLoc(mhist)
        if (loc.maxVal > 0) mhist.convertTo(mhist, -1, 1.0f / loc.maxVal, 0.0)

        return mhist

    }

    fun getFeatures(inMat: Mat, sizeData: Double): Mat {
        val VERTICAL = 1
        val HORIZONTAL = 2

        val vhist = getProjectedHistogram(inMat, VERTICAL)
        val hhist = getProjectedHistogram(inMat, HORIZONTAL)

        val lowData = Mat()
        resize(inMat, lowData, Size(sizeData, sizeData))

        val numCols = vhist.cols() + hhist.cols() + lowData.cols() * lowData.cols()
        val output = Mat.zeros(1, numCols, CV_32F)

        var j = 0;

        (0 until vhist.cols()).forEach {
            output.put(0, j, vhist.get(0, it).map { it.toFloat() }.toFloatArray())
            j++
        }

        (0 until hhist.cols()).forEach {
            output.put(0, j, hhist.get(0, it).map { it.toFloat() }.toFloatArray())
            j++
        }

        (0 until lowData.cols()).forEach {
            (0 until lowData.rows()).forEach {
                output.put(0, j, lowData.get(0, it).map { it.toFloat() }.toFloatArray())
            }
        }

        return output

    }

    fun verifySizeForChar(r: Mat): Boolean {

        val aspect = 45.0f / 77.0f;
        val charAspect = r.cols() / r.rows();
        val error = 0.35;
        val minHeight = 15;
        val maxHeight = 28;

        //We have a different aspect ratio for number 1, and it can be ~0.2
        val minAspect = 0.2;
        val maxAspect = aspect + aspect * error;

        //area of pixels
        val area = countNonZero(r);

        //bb area
        val bbArea = r.cols() * r.rows();

        //% of pixel in area
        val percPixels = area / bbArea;
        return percPixels < 0.8 && charAspect > minAspect && charAspect <
                maxAspect && r.rows() >= minHeight && r.rows() < maxHeight

    }

    var time: Long = System.currentTimeMillis();

    private fun getTimeDiff(): Long {
        val diff = System.currentTimeMillis() - time;
        time = System.currentTimeMillis()
        return diff
    }

    private fun initTimer() {
        time = System.currentTimeMillis()
    }


    private fun getPlateCandidatesFromImage(input: Mat, result: Mat, rects: MutableList<RotatedRect>): List<Plate> {

        val output = mutableListOf<Plate>()

        Log.i(TAG, " ${rects.size} rects found")

        rects.forEach { rect ->

            initTimer()

            Log.i(TAG, "   2-1) ${getTimeDiff()}")
            //For better rect cropping for each possible box
            //Make floodfill algorithm because the plate has white background
            //And then we can retrieve more clearly the contour box
            circle(result, rect.center, 3, Scalar(0.0, 255.0, 0.0), -1);

            val minSize = if (rect.size.width < rect.size.height) rect.size.width * 0.5 else rect.size.height * 0.5

            var mask = Mat()
            mask.create(input.rows() + 2, input.cols() + 2, CvType.CV_8UC1)
            mask = Mat.zeros(mask.size(), CvType.CV_8UC1)

            val loDiff = 30.0
            val upDiff = 30.0
            val connectivity = 4
            val newMaskVal = 255
            val seedNum = 10
            val ccomp: Rect = Rect()
            val flags = connectivity + (newMaskVal.shl(8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY)

            (0..seedNum).forEach { sn ->
                val num = Random().nextInt()
                val seed: Point = Point()
                seed.x = rect.center.x + num % (minSize - (minSize / 2))
                seed.y = rect.center.y + num % (minSize - (minSize / 2))
                circle(result, seed, 1, Scalar(0.0, 255.0, 255.0), -1);
                val area = floodFill(input, mask, seed, Scalar(255.0, 0.0, 0.0), ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags)
            }

            Log.i(TAG, "   2-2) ${getTimeDiff()} after FLOODFILL ")

            //Check new floodfill mask match for a correct patch.
            //Get all points detected for get Minimal rotated Rect
//            val pointsInterestList: MutableList<Point> = mutableListOf()

            val pointsInterestList: MutableList<Point> = arrayListOf()

            (0 until mask.cols()).forEach { col ->
                (0 until mask.rows()).forEach { row ->
                    if (mask.get(row, col)[0] == 255.0) {
                        pointsInterestList.add(Point(col.toDouble(), row.toDouble()))
                    }
                }
            }

            Log.i(TAG, "   2-3) ${getTimeDiff()} after MASKING ")

            val m2fFromList = MatOfPoint2f()
            m2fFromList.fromList(pointsInterestList) //create MatOfPoint2f from list of points
            val m2f = MatOfPoint2f()
            m2fFromList.convertTo(m2f, CvType.CV_32FC2) //convert to type of MatOfPoint2f created from list of points

            val minRect = Imgproc.minAreaRect(m2fFromList)

            if (verifySizes(minRect)) {

                // rotated rectangle drawing
                val rectPoints: Array<Point> = arrayOf<Point>(Point(), Point(), Point(), Point())
//            val rectPoints = MatOfPoint2f().toArray()

                minRect.points(rectPoints)

                (0 until 4).forEach { line(result, rectPoints[it], rectPoints[(it + 1) % 4], Scalar(0.0, 0.0, 255.0), 1) }

                val r = minRect.size.width / minRect.size.height
                var angle = minRect.angle
                if (r < 1) angle = 90 + angle

                val rotatedMat = Mat()
                warpAffine(input, rotatedMat, getRotationMatrix2D(minRect.center, angle, 1.0), input.size(), INTER_CUBIC) //TODO: 이거 같은가 봐야함.

                Log.i(TAG, "   2-4) ${getTimeDiff()} after WARP AFFINE ")

                val rectSize = minRect.size
                if (r < 1) {
                    val h = rectSize.height
                    val w = rectSize.width
                    rectSize.height = w
                    rectSize.width = h
                }

                val cropMat = Mat()
                getRectSubPix(rotatedMat, rectSize, minRect.center, cropMat)

                val resizedResultMat = Mat()
                resizedResultMat.create(33, 144, CV_8UC3)
                resize(cropMat, resizedResultMat, resizedResultMat.size(), 0.0, 0.0, INTER_CUBIC)

                //Equalized cropped image
                var grayResultMat = Mat()
                cvtColor(resizedResultMat, grayResultMat, COLOR_BGR2GRAY) //TODO: 상수확인
                blur(grayResultMat, grayResultMat, Size(3.0, 3.0))
                grayResultMat = histeq(grayResultMat)

                Log.i(TAG, "   2-5) ${getTimeDiff()} after EQUALIZING CROP ")

                //Plate Candidates Here
                output.add(Plate(grayResultMat, minRect.boundingRect(), ""))

            }

        }

        return output

    }

    private fun histeq(input: Mat): Mat {
        val output = Mat(input.size(), input.type())
        when (input.channels()) {
            3 -> {
                val hsv = Mat()
                val hsvSplit: List<Mat> = emptyList()
                cvtColor(input, hsv, COLOR_BGR2HSV)
                split(hsv, hsvSplit)
                equalizeHist(hsvSplit[2], hsvSplit[2])
                merge(hsvSplit, hsv)
                cvtColor(hsv, output, COLOR_HSV2BGR)
            }
            1 -> equalizeHist(input, output)
        }
        return output
    }


    private fun verifySizes(candidate: RotatedRect): Boolean {

        val error = 0.4
        val aspect = 4.7272
        val min = 15 * aspect * 15
        val max = 125 * aspect * 125
        val rmin = aspect - aspect * error
        val rmax = aspect + aspect * error
        val area = candidate.size.height * candidate.size.width;

        var r = candidate.size.width / candidate.size.height
        if (r < 1) r = 1 / r

        return !((area < min || area > max) || (r < rmin || r > rmax))

    }

    external fun salt(matAddrGray: Long, nbrElem: Int)

    companion object {

        private val TAG = "OCVSample::Activity"
    }
}

