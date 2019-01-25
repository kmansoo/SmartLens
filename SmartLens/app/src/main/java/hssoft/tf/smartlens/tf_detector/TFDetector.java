package hssoft.tf.smartlens.tf_detector;

import android.content.Context;
import android.graphics.Bitmap;

import java.io.IOException;
import java.util.List;

public interface TFDetector {
  boolean loadModel(Context context) throws IOException;
  boolean close();

  List<TFRecognition> recognizeImage(final Bitmap bitmap);
}
