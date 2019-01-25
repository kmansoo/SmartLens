package hssoft.tf.smartlens.tf_mobile;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import android.content.Context;
import android.graphics.Bitmap;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

import hssoft.tf.smartlens.tf_detector.TFDetector;
import hssoft.tf.smartlens.tf_detector.TFImageHelper;
import hssoft.tf.smartlens.tf_detector.TFRecognition;

public class TFMobileDetector implements TFDetector {
  public static final int INPUT_SIZE = 224;

  private static final float THRESHOLD = 0.1f;
  private static final int MAX_RESULTS = 3;
  private static final String MODEL_FILE = "mobilenet_v1.pb";
  private static final String LABEL_FILE = "labels.txt";
  private static final int CLASS_SIZE = 1001;
  private static final String INPUT_NAME = "input";
  private static final String OUTPUT_NAME = "MobilenetV1/Predictions/Reshape_1";
  private static final String[] OUTPUT_NAMES = {OUTPUT_NAME};

  private Context context_ = null;
  private Operation output_op;
  private TensorFlowInferenceInterface tfInterface_ = null;
  private Vector<String> labels_;

  public TFMobileDetector() {
  }

  // Implements TFDetector interfaces
  public boolean loadModel(Context context) {
    this.context_ = context;

    this.tfInterface_ = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILE);

    if (this.tfInterface_ == null)
      return false;

    return initLabels();
  }

  public boolean close() {
    return true;
  }

  public List<TFRecognition> recognizeImage(final Bitmap bitmap) {
    return  recognizeImage(TFImageHelper.bitmapToFloat(bitmap));
  }

  // Implements private methods
  private boolean initLabels() {
    boolean result = true;

    labels_ = new Vector<>(CLASS_SIZE);

    try {
      BufferedReader br = null;
      InputStream stream  = context_.getAssets().open(LABEL_FILE);
      br = new BufferedReader(new InputStreamReader(stream));
      String line;

      while ((line = br.readLine()) != null)
        labels_.add(line);

      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!" , e);
    }

    return result;
  }

  private List<TFRecognition> recognizeImage(final float[] imageFloats) {

    this.tfInterface_.feed(INPUT_NAME, imageFloats, 1, INPUT_SIZE, INPUT_SIZE, 3);

    this.tfInterface_.run(OUTPUT_NAMES, false);

    float[] outputs = new float[CLASS_SIZE];
    this.tfInterface_.fetch(OUTPUT_NAME, outputs);

    PriorityQueue<TFRecognition> pq =
        new PriorityQueue<>(
            3,
            new Comparator<TFRecognition>() {
              @Override
              public int compare(TFRecognition lhs, TFRecognition rhs) {
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });

    for (int i = 0; i < outputs.length; ++i) {
      if (outputs[i] > THRESHOLD) {
        pq.add(new TFRecognition("" + i, labels_.get(i), outputs[i], null));
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
