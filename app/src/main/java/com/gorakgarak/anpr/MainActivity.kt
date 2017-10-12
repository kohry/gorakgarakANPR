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
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc



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


        return matThreshold
    }

    external fun salt(matAddrGray: Long, nbrElem: Int)

    companion object {

        private val TAG = "OCVSample::Activity"
    }
}

