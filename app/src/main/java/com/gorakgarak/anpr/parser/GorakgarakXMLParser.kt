package com.gorakgarak.anpr.parser

import org.xmlpull.v1.XmlPullParser
import android.util.Xml
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.xmlpull.v1.XmlPullParserException
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.util.*

/**
 * Created by kohry on 2017-10-22.
 */
object GorakgarakXMLParser {

    private val namespace: String? = null

    @Throws(XmlPullParserException::class, IOException::class)
    fun parse(inputStream: InputStream, trainingDataTagName: String): Pair<Mat, Mat> {
        try {
            val parser = Xml.newPullParser()
            parser.setFeature(XmlPullParser.FEATURE_PROCESS_NAMESPACES, false)
            parser.setInput(inputStream, null)
            parser.nextTag()
            parser.require(XmlPullParser.START_TAG, namespace, "opencv_storage")
            return Pair(readFeed(parser, trainingDataTagName), readFeed(parser, "classes"))
        } finally {
            inputStream.close()
        }
    }

    @Throws(XmlPullParserException::class, IOException::class)
    private fun readFeed(parser: XmlPullParser, tagName:String): Mat {

        var mat = Mat() //you have to initialize opencv first, before loading this.

        while (parser.next() != XmlPullParser.END_TAG) {
            if (parser.eventType != XmlPullParser.START_TAG) {
                continue
            }
            val name = parser.name
            // Starts by looking for the entry tag
            if (name == tagName) {
                mat = readEntry(parser, tagName)
                break
            } else {
                skip(parser)
            }
        }

        return mat
    }

    @Throws(XmlPullParserException::class, IOException::class)
    private fun readEntry(parser: XmlPullParser, entryName: String): Mat {
        parser.require(XmlPullParser.START_TAG, namespace, entryName)
        var rows = 0
        var cols = 0
        var dt = ""
        var data = ""
        while (parser.next() != XmlPullParser.END_TAG) {
            if (parser.eventType != XmlPullParser.START_TAG) {
                continue
            }
            val name = parser.name
            when (name) {
                "rows" -> rows = readNode(parser, "rows").toInt()
                "cols" -> cols = readNode(parser, "cols").toInt()
                "dt" -> dt = readNode(parser, "dt")
                "data" ->  { data = readNode(parser, "data") }
                else -> skip(parser)
            }
        }

        var imageType = CvType.CV_32F
        if (dt == "f") imageType = CvType.CV_32F else if(dt == "i") imageType = CvType.CV_32S

        val mat = Mat(rows, cols, imageType)

        val st = StringTokenizer(data)

        (0 until rows).forEach { row ->
            (0 until cols).forEach { col ->
                if (dt == "f") mat.put(row, col, floatArrayOf(st.nextToken().toFloat()))
                else if (dt == "i") mat.put(row, col, intArrayOf(st.nextToken().toInt()))
            }
        }

        return mat
    }


    @Throws(IOException::class, XmlPullParserException::class)
    private fun readNode(parser: XmlPullParser, name: String): String {
        parser.require(XmlPullParser.START_TAG, namespace, name)
        val title = readText(parser)
        parser.require(XmlPullParser.END_TAG, namespace, name)
        return title
    }

    @Throws(IOException::class, XmlPullParserException::class)
    private fun readText(parser: XmlPullParser): String {
        var result = ""
        if (parser.next() == XmlPullParser.TEXT) {
            result = parser.text
            parser.nextTag()
        }
        return result
    }


    @Throws(XmlPullParserException::class, IOException::class)
    private fun skip(parser: XmlPullParser) {
        if (parser.eventType != XmlPullParser.START_TAG) {
            throw IllegalStateException()
        }
        var depth = 1
        while (depth != 0) {
            when (parser.next()) {
                XmlPullParser.END_TAG -> depth--
                XmlPullParser.START_TAG -> depth++
            }
        }
    }
}