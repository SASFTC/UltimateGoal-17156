package org.firstinspires.ftc.team17156;

import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.firstinspires.ftc.robotcore.internal.camera.WebcamExample;
import org.opencv.android.Utils;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.*;

import com.qualcomm.robotcore.hardware.HardwareMap;
import org.opencv.core.*;
import org.openftc.easyopencv.OpenCvInternalCamera;
import org.pytorch.*;
import org.pytorch.torchvision.TensorImageUtils;

import java.nio.file.Paths;
import java.util.Arrays;


@TeleOp(name = "PyTorch Ring Detection", group = "CV")
public class RingDetectionModel extends LinearOpMode
{
    OpenCvInternalCamera webcam;

    @Override
    public void runOpMode()
    {
        int cameraMonitorViewId = hardwareMap.appContext.getResources().getIdentifier("cameraMonitorViewId", "id", hardwareMap.appContext.getPackageName());
        webcam = OpenCvCameraFactory.getInstance().createInternalCamera(OpenCvInternalCamera.CameraDirection.BACK, cameraMonitorViewId);
        webcam.openCameraDevice();

        RingDetector detector = new RingDetector();
        webcam.setPipeline(detector);
        webcam.openCameraDeviceAsync(new OpenCvCamera.AsyncCameraOpenListener()
        {
            @Override
            public void onOpened()
            {
                /*
                 * Tell the webcam to start streaming images to us! Note that you must make sure
                 * the resolution you specify is supported by the camera. If it is not, an exception
                 * will be thrown.
                 *
                 * Keep in mind that the SDK's UVC driver (what OpenCvWebcam uses under the hood) only
                 * supports streaming from the webcam in the uncompressed YUV image format. This means
                 * that the maximum resolution you can stream at and still get up to 30FPS is 480p (640x480).
                 * Streaming at e.g. 720p will limit you to up to 10FPS and so on and so forth.
                 *
                 * Also, we specify the rotation that the webcam is used in. This is so that the image
                 * from the camera sensor can be rotated such that it is always displayed with the image upright.
                 * For a front facing camera, rotation is defined assuming the user is looking at the screen.
                 * For a rear facing camera or a webcam, rotation is defined assuming the camera is facing
                 * away from the user.
                 */
                webcam.startStreaming(320, 240, OpenCvCameraRotation.UPRIGHT);
            }
        });

        telemetry.addLine("Waiting for start");
        telemetry.update();

        /*
         * Wait for the user to press start on the Driver Station
         */
        waitForStart();

        while (opModeIsActive())
        {
            /*
             * Send some stats to the telemetry
             */
            telemetry.addData("Frame Count", webcam.getFrameCount());
            telemetry.addData("FPS", String.format("%.2f", webcam.getFps()));
            telemetry.addData("Total frame time ms", webcam.getTotalFrameTimeMs());
            telemetry.addData("Pipeline time ms", webcam.getPipelineTimeMs());
            telemetry.addData("Overhead time ms", webcam.getOverheadTimeMs());
            telemetry.addData("Theoretical max FPS", webcam.getCurrentPipelineMaxFps());
            telemetry.update();

            if(gamepad1.a)
            {
                webcam.stopStreaming();
                webcam.closeCameraDevice(); // Optional
            }
            sleep(100);
        }
    }

    class RingDetector extends OpenCvPipeline {
        @RequiresApi(api = Build.VERSION_CODES.O)
        @Override
        public Mat processFrame(Mat input) {
            telemetry.addData("Processing Frame", "...");
            Module module = null;
            try {
                 Log.d("asd", Paths.get("").toString());
                 module = Module.load("/FIRST/rings.pt");
            } catch (Exception ignored) {
                return null;
            }
            Bitmap bitmap = null;

            Mat tmp = new Mat (input.height(), input.width(), CvType.CV_8U, new Scalar(4));
            try {
                //Imgproc.cvtColor(seedsImage, tmp, Imgproc.COLOR_RGB2BGRA);
                Imgproc.cvtColor(input, tmp, Imgproc.COLOR_GRAY2RGBA, 4);
                bitmap = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(tmp, bitmap);
            }
            catch (Exception e){
                telemetry.addData("FATAL:",e.getMessage());
                return null;
            }

            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

            Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
            float[] scores = outputTensor.getDataAsFloatArray();
            telemetry.addData("RES:", Arrays.toString(outputTensor.shape()));

            telemetry.addData("Detection Result: ", scores[0]);

            // Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor()
            return input; // FIXME
        }
    }
}
