package hssoft.tf.smartlens.tf_lite;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

// TF Lite와 결과를 통합
import hssoft.tf.smartlens.TFRecognition;

/**
 * Created by amitshekhar on 17/03/18.
 */

public class TFLiteImageClassifier implements TFLiteClassifier {

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    private Interpreter interpreter;
    private int inputSize;
    private List<String> labelList;

    private TFLiteImageClassifier() {

    }

    public static TFLiteImageClassifier create(Context context,
                                               String modelPath,
                                               String labelPath,
                                               int inputSize) throws IOException {

        Interpreter.Options tfliteOptions = null;

        if (TFLiteGpuDelegateHelper.isGpuDelegateAvailable()) {
            tfliteOptions = new Interpreter.Options();

            Delegate gpuDelegate = TFLiteGpuDelegateHelper.createGpuDelegate();

            tfliteOptions.addDelegate(gpuDelegate);
        }

        TFLiteImageClassifier classifier = new TFLiteImageClassifier();

        classifier.labelList = classifier.loadLabelList(context.getAssets(), labelPath);

        if (tfliteOptions == null)
            classifier.interpreter = new Interpreter(classifier.loadModelFile(context.getAssets(), modelPath));
        else
            classifier.interpreter = new Interpreter(classifier.loadModelFile(context.getAssets(), modelPath), tfliteOptions);

        classifier.inputSize = inputSize;

        return classifier;
    }

    @Override
    public List<TFRecognition> recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        byte[][] result = new byte[1][labelList.size()];
        interpreter.run(byteBuffer, result);
        return getSortedResult(result);
    }

    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.put((byte) ((val >> 16) & 0xFF));
                byteBuffer.put((byte) ((val >> 8) & 0xFF));
                byteBuffer.put((byte) (val & 0xFF));
            }
        }
        return byteBuffer;
    }

    @SuppressLint("DefaultLocale")
    private List<TFRecognition> getSortedResult(byte[][] labelProbArray) {

        PriorityQueue<TFRecognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<TFRecognition>() {
                            @Override
                            public int compare(TFRecognition lhs, TFRecognition rhs) {
                                return Float.compare(lhs.getConfidence(), rhs.getConfidence());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = (labelProbArray[0][i] & 0xff) / 255.0f;
            if (confidence > THRESHOLD) {
                pq.add(new TFRecognition("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence,
                        null));
            }
        }

        final ArrayList<TFRecognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

}
