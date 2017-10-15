package com.gorakgarak.anpr.utils

import org.opencv.core.Mat

/**
 * Created by kohry on 2017-10-15.
 */
object CustomImageProc {

    fun setPixel(mat: Mat, index:Int, value:Float) {
        val buff = floatArrayOf()
        mat.get(0, 0, buff)
        buff[index] = value
        mat.put(0, 0, buff)
    }


    fun setPixel(mat: Mat, index:Int, value:Byte) {
        val buff = byteArrayOf()
        mat.get(0, 0, buff)
        buff[index] = value
        mat.put(0, 0, buff)
    }

    fun setPixel(mat: Mat, index:Int, value:Int) {
        val buff = intArrayOf()
        mat.get(0, 0, buff)
        buff[index] = value
        mat.put(0, 0, buff)
    }


}