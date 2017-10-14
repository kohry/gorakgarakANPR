
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
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.core.CvType.CV_8UC1
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.*
import java.util.*
import org.opencv.core.CvType
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint




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
        val contourList: List<MatOfPoint> = emptyList()
        Imgproc.findContours(matThreshold, contourList, null, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)

        Log.d(TAG, "1-6) Get rectangle from the contours")
        val rectList: MutableList<RotatedRect> = mutableListOf()

        //Convert MapOfPoint to MatOfPoint2f
        contourList.forEach {
            val p = MatOfPoint2f()
            it.convertTo(p, CvType.CV_32F)
            rectList.add(Imgproc.minAreaRect(p))
        }

        val input = inputFrame.rgba()
        val result = Mat()
        input.copyTo(result)
        drawContours(result, contourList, -1, Scalar(200.0, 0.0, 0.0), 1)

        Log.d(TAG, "1-6) Floodfill algorithm from more clear contour box")
        floodFill(input, result, rectList)

        return result
    }

    fun floodFill(input: Mat, result:Mat, rects: MutableList<RotatedRect>) {

        rects.forEach {
            //For better rect cropping for each possible box
            //Make floodfill algorithm because the plate has white background
            //And then we can retrieve more clearly the contour box
            circle(result, it.center, 3, Scalar(0.0,255.0,0.0), -1);

            val minSize = if (it.size.width < it.size.height) it.size.width * 0.5 else it.size.height * 0.5

            var mask: Mat = Mat()
            mask.create(input.rows() + 2, input.cols() + 2, CvType.CV_8UC1)
            mask = Mat.zeros(mask.size(), CvType.CV_8UC3)

            val loDiff = 30.0
            val upDiff = 30.0
            val connectivity = 4
            val newMaskVal = 255
            val seedNum = 10
            val ccomp : Rect = Rect()
            val flags = connectivity + (newMaskVal.shl(8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY)

            (0 .. seedNum).forEach {
                val num = Random().nextInt()
                val seed:Point = Point()
                seed.x=rects.get(it).center.x + num % (minSize-(minSize/2))
                seed.y=rects.get(it).center.y + num % (minSize-(minSize/2))
                circle(result, seed, 1, Scalar(0.0,255.0,255.0), -1);
                val area = floodFill(input, mask, seed, Scalar(255.0, 0.0, 0.0), ccomp , Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags)
            }

//            val pointsOfInterest: List<Point> = mutableListOf()

//            vector<Point> pointsInterest;
//            Mat_<uchar>::iterator itMask= mask.begin<uchar>();
//            Mat_<uchar>::iterator end= mask.end<uchar>();
//            for( ; itMask!=end; ++itMask)
//            if(*itMask==255)
//            pointsInterest.push_back(itMask.pos());

        }



    }


    fun verifySizes(candidate: RotatedRect): Boolean {

        val error = 0.4
        val aspect = 4.7272
        val min = 15 * aspect * 15
        val max = 125 * aspect * 125
        val rmin = aspect - aspect * error
        val rmax = aspect + aspect * error
        val area = candidate.size.height * candidate.size.width;

        var r = candidate.size.width / candidate.size.height
        if (r < 1) r = 1 / r

        return !(( area < min || area > max ) || ( r < rmin || r > rmax ))

    }

    external fun salt(matAddrGray: Long, nbrElem: Int)

    companion object {

        private val TAG = "OCVSample::Activity"
    }
}

