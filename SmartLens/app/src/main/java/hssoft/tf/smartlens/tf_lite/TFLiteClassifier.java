package hssoft.tf.smartlens.tf_lite;

import android.graphics.Bitmap;

import java.util.List;

// TF Lite와 결과를 통합
import hssoft.tf.smartlens.TFRecognition;

/**
 * Created by amitshekhar on 17/03/18.
 */

public interface TFLiteClassifier {

    List<TFRecognition> recognizeImage(Bitmap bitmap);

    void close();
}
