package com.example.aslaiapp;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.ForegroundColorSpan;
import android.util.Log;

import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsOptions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.StringTokenizer;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

/** Classifies images with Tensorflow Lite. */
public class ImageClassifier {
    /** Tag for the {@link Log}. */
    private static final String TAG = "TfLiteCameraDemo";

    /** Number of results to show in the UI. */
    private static final int RESULTS_TO_SHOW = 3;

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /** Labels corresponding to the output of the vision model. */
    private List<String> labelList;

    protected float[] handData = null;

    protected boolean noHands = false;

    /** Options for configuring the Interpreter.*/
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** An instance of host activity.*/
    private Activity activity = null;

    /** multi-stage low pass filter * */
    private float[][] filterLabelProbArray = null;

    private static final int FILTER_STAGES = 3;
    private static final float FILTER_FACTOR = 0.4f;

    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
     * of the super class, because we need a primitive array here.
     */
    private float[][] labelProbArray = null;

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    private Hands handFinder = null;

    /** Initializes an {@code ImageClassifier}. */
    ImageClassifier(Activity activity) throws IOException {
        this.activity = activity;
        initializeModel();
        labelList = loadLabelList();

        handData = new float[63];

        filterLabelProbArray = new float[FILTER_STAGES][getNumLabels()];
        labelProbArray = new float[1][getNumLabels()];

        HandsOptions handsOptions = HandsOptions.builder()
                .setStaticImageMode(false)
                .setMaxNumHands(2)
                .setRunOnGpu(true).build();
        handFinder = new Hands(activity, handsOptions);
        handFinder.setErrorListener(
                (message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));
        handFinder.setResultListener(
                handsResult -> {
                    if (handsResult.multiHandLandmarks().isEmpty()) {
                        noHands = true;
                        return;
                    }
                    noHands = false;
                    List<LandmarkProto.NormalizedLandmark> list =
                            handsResult.multiHandLandmarks().get(0).getLandmarkList();
                    for (int i = 0; i < 21; i++) {
                        LandmarkProto.NormalizedLandmark landmark = list.get(i);
                        handData[i*3] = landmark.getX();
                        handData[i*3+1] = landmark.getY();
                        handData[i*3+2] = landmark.getZ();
                    }
                });

        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
    }

    /** Classifies a frame from the preview stream. */
    void classifyFrame(Bitmap bitmap, SpannableStringBuilder builder) {
        printTopKLabels(builder);

        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            builder.append(new SpannableString("Uninitialized Classifier."));
        }

        handFinder.send(bitmap, System.currentTimeMillis());
        if (noHands) {
            return;
        }

        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();
        runInference();
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + (endTime - startTime));

        // Smooth the results across frames.
        applyFilter();

        // Print the results.
        long duration = endTime - startTime;
        SpannableString span = new SpannableString(duration + " ms");
        span.setSpan(new ForegroundColorSpan(android.graphics.Color.LTGRAY), 0, span.length(), 0);
        builder.append(span);
    }

    void applyFilter() {
        int numLabels = getNumLabels();

        // Low pass filter `labelProbArray` into the first stage of the filter.
        for (int j = 0; j < numLabels; ++j) {
            filterLabelProbArray[0][j] +=
                    FILTER_FACTOR * (getProbability(j) - filterLabelProbArray[0][j]);
        }
        // Low pass filter each stage into the next.
        for (int i = 1; i < FILTER_STAGES; ++i) {
            for (int j = 0; j < numLabels; ++j) {
                filterLabelProbArray[i][j] +=
                        FILTER_FACTOR * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j]);
            }
        }

        // Copy the last stage filter output back to `labelProbArray`.
        for (int j = 0; j < numLabels; ++j) {
            setProbability(j, filterLabelProbArray[FILTER_STAGES - 1][j]);
        }
    }

    /* Sets number of threads and re-initialize model. */
    public void setNumThreads(int numThreads) throws IOException {
        if (tflite != null) {
            tfliteOptions.setNumThreads(numThreads);
            close();
        }
        initializeModel();
    }

    /** Closes tflite to release resources. */
    public void close() {
        tflite.close();
        tflite = null;
    }

    private void initializeModel() throws IOException {
        if (tflite == null) {
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
            tflite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    /** Reads label list from Assets. */
    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())));
        String line;
        line = reader.readLine();
        reader.close();

        StringTokenizer tokenizer = new StringTokenizer(line, ",");
        while (tokenizer.hasMoreTokens()) {
            String token = tokenizer.nextToken();
            labelList.add(token);
        }
        return labelList;
    }

    /** Prints top-K labels, to be shown in UI as the results. */
    private void printTopKLabels(SpannableStringBuilder builder) {
        for (int i = 0; i < getNumLabels(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), getNormalizedProbability(i)));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }

        final int size = sortedLabels.size();
        for (int i = 0; i < size; i++) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            SpannableString span =
                    new SpannableString(String.format("%s:  %4.2f\n", label.getKey(), label.getValue()));
            builder.insert(0, span);
        }
    }

    /**
     * Get the name of the model file stored in Assets.
     *
     * @return string
     */
    protected String getModelPath() {
        // you can download this file from
        return "model.tflite";
    }

    /**
     * Get the name of the label file stored in Assets.
     *
     * @return string
     */
    protected String getLabelPath() {
        return "labels.txt";
    }

    /**
     * Read the probability value for the specified label This is either the original value as it was
     * read from the net's output or the updated value after the filter was applied.
     *
     * @param labelIndex int
     * @return float
     */
    protected float getProbability(int labelIndex) {
        return labelProbArray[0][labelIndex];
    }

    /**
     * Set the probability value for the specified label.
     *
     * @param labelIndex int
     * @param value Number
     */
    protected void setProbability(int labelIndex, Number value) {
        labelProbArray[0][labelIndex] = value.floatValue();
    }

    /**
     * Get the normalized probability value for the specified label. This is the final value as it
     * will be shown to the user.
     *
     * @return float
     */
    protected float getNormalizedProbability(int labelIndex) {
        return getProbability(labelIndex);
    }

    /**
     * Run inference using the prepared input in imgData. Afterwards, the result will be
     * provided by getProbability().
     *
     * <p>This additional method is necessary, because we don't have a common base for different
     * primitive data types.
     */
    protected void runInference() {
        tflite.run(handData, labelProbArray);
    }

    /**
     * Get the total number of labels.
     *
     * @return int
     */
    protected int getNumLabels() {
        return labelList.size();
    }
}
