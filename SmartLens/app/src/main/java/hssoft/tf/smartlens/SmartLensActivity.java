package hssoft.tf.smartlens;

import android.Manifest;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Display;
import android.view.View;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Toast;

import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import hssoft.tf.smartlens.tf_lite.TFLiteImageClassifier;
import hssoft.tf.smartlens.tf_mobile.TFMobileImageClassifier;
import hssoft.tf.smartlens.tf_mobile.TFMobileImageHelper;
import hssoft.tf.smartlens.views.CameraPreviewFragment;

/**
 * An example full-screen activity that shows and hides the system UI (i.e.
 * status bar and navigation/system bar) with user interaction.
 */
public class SmartLensActivity
        extends AppCompatActivity
        implements CameraPreviewFragment.CameraPreviewListener {

    private static final String TAG = "FullscreenActivity";

    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final int PERMISSIONS_REQUEST = 1;

    private static final boolean DEBUG_MODE = false;

    private ListView resultListView;
    private ImageView previewImageView;
    private Button btnTFMethod;


    private boolean computing;
    private Bitmap bitmap;
    private Bitmap croppedBitmap;
    private int sensorOrientation;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    // ++
    private static final int TF_METHOD_TFLITE = 0;
    private static final int TF_METHOD_TFMOBILE = 1;

    private int tfMethod = TF_METHOD_TFMOBILE;

    //  TF Mobile
    private TFMobileImageClassifier tfMobileImageClassifier;
    private ArrayAdapter<TFRecognition> resultListAdapter;
    private List<TFRecognition> tfMobileResultList;

    //  TF Lite
    private Executor executor = Executors.newSingleThreadExecutor();
    private TFLiteImageClassifier tfLiteImageClassifier;

    private static final String TFLITE_MODEL_PATH = "mobilenet_v1_1.0_224_quant.tflite";
    private static final String TFLITE_LABEL_PATH = "labels.txt";
    private static final int TFLITE_INPUT_SIZE = 224;

    private Runnable updateResult = new Runnable() {
        @Override
        public void run() {
            resultListAdapter.clear();

            if (tfMobileResultList != null) {
                resultListAdapter.addAll(tfMobileResultList);
            }

            if (DEBUG_MODE) {
                previewImageView.setImageDrawable(new BitmapDrawable(getResources(), croppedBitmap));
            }

            computing = false;
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_fullscreen);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (hasPermission()) {
            if (null == savedInstanceState) {
                init();
            }
        } else {
            requestPermission();
        }

        btnTFMethod = findViewById(R.id.tf_method);
        changeTFMethod();

        btnTFMethod.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                changeTFMethod();
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (tfLiteImageClassifier != null) {
            executor.execute(new Runnable() {
                @Override
                public void run() {
                    tfLiteImageClassifier.close();
                }
            });
            tfLiteImageClassifier = null;
        }
    }


    private void init() {
        resultListView = (ListView) findViewById(R.id.resultList);
        previewImageView = (ImageView) findViewById(R.id.previewImage);

        resultListAdapter = new ArrayAdapter<>(this, R.layout.item_recogition);
        resultListView.setAdapter(this.resultListAdapter);

        getFragmentManager().beginTransaction()
                .replace(R.id.container, CameraPreviewFragment.newInstance(this))
                .commit();


        initTFLite();
        initTFMobile();


        if (DEBUG_MODE) {
            previewImageView.setVisibility(View.VISIBLE);
        } else {
            previewImageView.setVisibility(View.GONE);
        }

    }

    private void changeTFMethod() {
        if (tfMethod == TF_METHOD_TFLITE) {
            tfMethod = TF_METHOD_TFMOBILE;

            btnTFMethod.setText("TF Mobile");
            btnTFMethod.setBackgroundColor(Color.BLUE);
        }
        else {
            tfMethod = TF_METHOD_TFLITE;

            btnTFMethod.setText("TF Lite");
            btnTFMethod.setBackgroundColor(Color.RED);
        }

    }

    private void initTFMobile() {
        tfMobileImageClassifier = new TFMobileImageClassifier(this);
    }

    private void initTFLite() {
        try {
            tfLiteImageClassifier = TFLiteImageClassifier.create(
                    this,
                    TFLITE_MODEL_PATH,
                    TFLITE_LABEL_PATH,
                    TFLITE_INPUT_SIZE);
        } catch (final Exception e) {
            throw new RuntimeException("Error initializing TensorFlow!", e);
        }

        /*
            executor.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        tfLiteImageClassifier = TFLiteImageClassifier.create(
                                getAssets(),
                                TFLITE_MODEL_PATH,
                                TFLITE_LABEL_PATH,
                                TFLITE_INPUT_SIZE);
                    } catch (final Exception e) {
                        throw new RuntimeException("Error initializing TensorFlow!", e);
                    }
                }
            });
        */
    }

    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
    }


    @Override
    public void onPreviewReadied(Size size, int cameraRotation) {
        bitmap = Bitmap.createBitmap(size.getWidth(), size.getHeight(), Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(TFMobileImageClassifier.INPUT_SIZE, TFMobileImageClassifier.INPUT_SIZE, Bitmap.Config.ARGB_8888);

        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();

        sensorOrientation = cameraRotation + screenOrientation;

        frameToCropTransform =
                TFMobileImageHelper.getTransformationMatrix(
                        size.getHeight(), size.getWidth(),
                        TFMobileImageClassifier.INPUT_SIZE, TFMobileImageClassifier.INPUT_SIZE,
                        sensorOrientation, true);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
    }

    @Override
    public void onImageAvailable(ImageReader reader) {
        if (computing)
            return;

        Image image = null;

        try {

            image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            computing = true;

            TFMobileImageHelper.imageToBitmap(image, bitmap);

            final Canvas canvas = new Canvas(croppedBitmap);
            canvas.drawBitmap(bitmap, frameToCropTransform, null);

            image.close();

            if (tfMethod == TF_METHOD_TFMOBILE) {
                //  TF Mobile
                tfMobileResultList = tfMobileImageClassifier.recognizeImage(croppedBitmap);

                //  Log.d(TAG, "recognizeImage using TF Mobile");
            }
            else {
                //  TF Lite
                tfMobileResultList = tfLiteImageClassifier.recognizeImage(croppedBitmap);

                //  Log.d(TAG, "recognizeImage using TF Lite");
            }

            runOnUiThread(updateResult);
        } catch (final Exception e) {
            Log.e(TAG, "recognizeImage", e);
        } finally {
            if (image != null) {
                image.close();
            }

            computing = false;
        }
    }

    //region Permission
    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            boolean result = checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;

            return result;

        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(PERMISSION_CAMERA) != PackageManager.PERMISSION_GRANTED) {
                if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                    Toast.makeText(SmartLensActivity.this, "Camera permission is required for this example", Toast.LENGTH_LONG).show();
                }

                requestPermissions(new String[]{PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
                return;
            }

            /*
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                if (shouldShowRequestPermissionRationale(Manifest.permission.READ_EXTERNAL_STORAGE)) {
                    Toast.makeText(SmartLensActivity.this, "READ_EXTERNAL_STORAGE permission is required for this example", Toast.LENGTH_LONG).show();
                }

                requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST);
                return;
            }

            if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                if (shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                    Toast.makeText(SmartLensActivity.this, "WRITE_EXTERNAL_STORAGE permission is required for this example", Toast.LENGTH_LONG).show();
                }

                requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST);
                return;
            }
            */
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST) {
            if (hasPermission()) {
                init();
            } else {
                requestPermission();
            }
        }
    }
}
