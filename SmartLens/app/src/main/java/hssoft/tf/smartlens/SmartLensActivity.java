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

import hssoft.tf.smartlens.tf_detector.TFDetector;
import hssoft.tf.smartlens.tf_detector.TFImageHelper;
import hssoft.tf.smartlens.tf_detector.TFRecognition;
import hssoft.tf.smartlens.tf_lite.TFLiteDetector;
import hssoft.tf.smartlens.tf_mobile.TFMobileDetector;
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

    private ListView resultListView_;
    private ImageView previewImageView_;
    private Button btnTFMethod_;


    private boolean computing_;
    private Bitmap bitmap_;
    private Bitmap croppedBitmap_;
    private int sensorOrientation_;
    private Matrix frameToCropTransform_;
    private Matrix cropToFrameTransform_;

    private static final int TF_METHOD_TFLITE = 0;
    private static final int TF_METHOD_TFMOBILE = 1;

    private int tfMethod_ = TF_METHOD_TFMOBILE;

    private ArrayAdapter<TFRecognition> resultListAdapter_;
    private List<TFRecognition> tfMobileResultList_;

    private TFDetector activiteDetector_;

    //  TF Mobile and Lite
    private TFDetector tfMobileDetector_  = null;
    private TFDetector tfLiteDetector_ = null;


    private Runnable updateResult = new Runnable() {
        @Override
        public void run() {
            resultListAdapter_.clear();

            if (tfMobileResultList_ != null)
                resultListAdapter_.addAll(tfMobileResultList_);

            if (DEBUG_MODE)
                previewImageView_.setImageDrawable(new BitmapDrawable(getResources(), croppedBitmap_));

            computing_ = false;
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_fullscreen);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        btnTFMethod_ = findViewById(R.id.tf_method);
        btnTFMethod_.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                changeTFMethod();
            }
        });

        if (hasPermission()) {
            if (null == savedInstanceState) {
                init();
            }
        } else {
            requestPermission();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        tfLiteDetector_ = null;
        tfMobileDetector_ = null;
    }

    private void init() {
        resultListView_ = (ListView) findViewById(R.id.resultList);
        previewImageView_ = (ImageView) findViewById(R.id.previewImage);

        resultListAdapter_ = new ArrayAdapter<>(this, R.layout.item_recogition);
        resultListView_.setAdapter(this.resultListAdapter_);

        getFragmentManager().beginTransaction()
            .replace(R.id.container, CameraPreviewFragment.newInstance(this))
            .commit();


        initTFLite();
        initTFMobile();

        changeTFMethod();

        if (DEBUG_MODE) {
            previewImageView_.setVisibility(View.VISIBLE);
        } else {
            previewImageView_.setVisibility(View.GONE);
        }

    }

    private void initTFMobile() {
        this.tfMobileDetector_ = new TFMobileDetector();

        try {
            this.tfMobileDetector_.loadModel(this);
        }
        catch (Exception e) {

        }
    }

    private void initTFLite() {
        this.tfLiteDetector_ = new TFLiteDetector();

        try {
            this.tfLiteDetector_.loadModel(this);
        }
        catch (Exception e) {
            throw new RuntimeException("Error initializing TensorFlow!", e);
        }
    }

    private void changeTFMethod() {
        if (tfMethod_ == TF_METHOD_TFLITE) {
            tfMethod_ = TF_METHOD_TFMOBILE;

            btnTFMethod_.setText("TF Mobile");
            btnTFMethod_.setBackgroundColor(Color.BLUE);

            activiteDetector_ = tfMobileDetector_;
        }
        else {
            tfMethod_ = TF_METHOD_TFLITE;

            btnTFMethod_.setText("TF Lite");
            btnTFMethod_.setBackgroundColor(Color.RED);

            activiteDetector_ = tfLiteDetector_;
        }
    }

    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
    }

    @Override
    public void onPreviewReadied(Size size, int cameraRotation) {
        bitmap_ = Bitmap.createBitmap(size.getWidth(), size.getHeight(), Bitmap.Config.ARGB_8888);
        croppedBitmap_ = Bitmap.createBitmap(TFMobileDetector.INPUT_SIZE, TFMobileDetector.INPUT_SIZE, Bitmap.Config.ARGB_8888);

        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();

        sensorOrientation_ = cameraRotation + screenOrientation;

        frameToCropTransform_ =
            TFImageHelper.getTransformationMatrix(
                size.getHeight(), size.getWidth(),
                TFMobileDetector.INPUT_SIZE, TFMobileDetector.INPUT_SIZE,
                sensorOrientation_, true);

        cropToFrameTransform_ = new Matrix();
        frameToCropTransform_.invert(cropToFrameTransform_);
    }

    @Override
    public void onImageAvailable(ImageReader reader) {
        if (computing_)
            return;

        Image image = null;

        try {

            image = reader.acquireLatestImage();

            if (image == null)
                return;

            computing_ = true;

            TFImageHelper.imageToBitmap(image, bitmap_);

            final Canvas canvas = new Canvas(croppedBitmap_);
            canvas.drawBitmap(bitmap_, frameToCropTransform_, null);

            image.close();

            if (activiteDetector_ != null)
                tfMobileResultList_ = activiteDetector_.recognizeImage(croppedBitmap_);

            runOnUiThread(updateResult);
        } catch (final Exception e) {
            Log.e(TAG, "recognizeImage", e);
        } finally {
            if (image != null) {
                image.close();
            }

            computing_ = false;
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
