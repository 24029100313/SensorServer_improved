package github.umer0586.sensorserver.activities

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.permissionx.guolindev.PermissionX
import github.umer0586.sensorserver.R
import github.umer0586.sensorserver.databinding.ActivityVideoStreamBinding
import github.umer0586.sensorserver.service.ServiceBindHelper
import github.umer0586.sensorserver.service.VideoServerStateListener
import github.umer0586.sensorserver.service.VideoStreamService
import github.umer0586.sensorserver.videostream.VideoServerInfo

class VideoStreamActivity : AppCompatActivity() {

    private val TAG = "VideoStreamActivity"
    private lateinit var binding: ActivityVideoStreamBinding
    
    private lateinit var videoServiceBindHelper: ServiceBindHelper
    private var videoService: VideoStreamService? = null
    
    private val CAMERA_PERMISSION_REQUEST_CODE = 100

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVideoStreamBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setSupportActionBar(binding.toolbar.root)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        
        // Setup service binding
        videoServiceBindHelper = ServiceBindHelper(
            context = applicationContext,
            service = VideoStreamService::class.java,
            componentLifecycle = lifecycle
        )
        
        videoServiceBindHelper.onServiceConnected(this::onVideoServiceConnected)
        
        // Setup UI
        binding.startStopButton.setOnClickListener {
            Log.d(TAG, "Start/Stop button clicked")
            val isServerRunning = videoService?.checkState() != null
            Log.d(TAG, "Is server running: $isServerRunning")
            
            if (!isServerRunning) {
                Log.d(TAG, "Attempting to start video service")
                if (checkCameraPermission()) {
                    Log.d(TAG, "Camera permission granted, starting service")
                    startVideoService()
                } else {
                    Log.d(TAG, "Camera permission not granted, requesting permission")
                    requestCameraPermission()
                }
            } else {
                Log.d(TAG, "Stopping video service")
                stopVideoService()
            }
        }
        
        binding.videoInfoTextView.text = "Video stream is not running"
    }
    
    private fun onVideoServiceConnected(binder: IBinder) {
        val localBinder = binder as VideoStreamService.LocalBinder
        videoService = localBinder.getService()
        
        videoService?.setServerStateListener(object : VideoServerStateListener {
            override fun onStart(videoServerInfo: VideoServerInfo) {
                runOnUiThread {
                    binding.startStopButton.text = "Stop Video Stream"
                    binding.videoInfoTextView.text = "Video stream running at:\nws://${videoServerInfo.ipAddress}:${videoServerInfo.port}/video"
                    binding.statusIndicator.setBackgroundResource(R.drawable.status_running)
                    Toast.makeText(this@VideoStreamActivity, "Video stream started", Toast.LENGTH_SHORT).show()
                }
            }
            
            override fun onStop() {
                runOnUiThread {
                    binding.startStopButton.text = "Start Video Stream"
                    binding.videoInfoTextView.text = "Video stream is not running"
                    binding.statusIndicator.setBackgroundResource(R.drawable.status_stopped)
                    Toast.makeText(this@VideoStreamActivity, "Video stream stopped", Toast.LENGTH_SHORT).show()
                }
            }
            
            override fun onError(exception: Exception?) {
                runOnUiThread {
                    binding.startStopButton.text = "Start Video Stream"
                    binding.videoInfoTextView.text = "Error: ${exception?.message}"
                    binding.statusIndicator.setBackgroundResource(R.drawable.status_error)
                    Toast.makeText(this@VideoStreamActivity, "Error: ${exception?.message}", Toast.LENGTH_SHORT).show()
                }
            }
            
            override fun onRunning(videoServerInfo: VideoServerInfo) {
                runOnUiThread {
                    binding.startStopButton.text = "Stop Video Stream"
                    binding.videoInfoTextView.text = "Video stream running at:\nws://${videoServerInfo.ipAddress}:${videoServerInfo.port}/video"
                    binding.statusIndicator.setBackgroundResource(R.drawable.status_running)
                }
            }
        })
        
        // Check current state
        videoService?.checkState()
    }
    
    private fun startVideoService() {
        Log.d(TAG, "Starting video service...")
        // Whether user grant this permission or not we will start service anyway
        // If permission is not granted foreground notification will not be shown
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Log.d(TAG, "Requesting POST_NOTIFICATIONS permission for Android 13+")
            PermissionX.init(this)
                .permissions(android.Manifest.permission.POST_NOTIFICATIONS)
                .request { allGranted, _, _ -> 
                    Log.d(TAG, "Notification permission request completed, granted: $allGranted")
                }
        }
        
        try {
            val intent = Intent(applicationContext, VideoStreamService::class.java)
            Log.d(TAG, "Creating intent for VideoStreamService: $intent")
            Log.d(TAG, "Calling startForegroundService")
            ContextCompat.startForegroundService(applicationContext, intent)
            Log.d(TAG, "startForegroundService called successfully")
            Toast.makeText(this, "Starting video stream service...", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Error starting service: ${e.message}", e)
            Toast.makeText(this, "Error starting service: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
    
    private fun stopVideoService() {
        val intent = Intent(VideoStreamService.ACTION_STOP_VIDEO_STREAM).apply {
            setPackage(applicationContext.packageName)
        }
        this.sendBroadcast(intent)
    }
    
    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun requestCameraPermission() {
        Log.d(TAG, "Requesting camera permission")
        PermissionX.init(this)
            .permissions(Manifest.permission.CAMERA)
            .request { allGranted, _, deniedList ->
                Log.d(TAG, "Camera permission request result: granted=$allGranted, denied=$deniedList")
                if (allGranted) {
                    Log.d(TAG, "Camera permission granted, starting video service")
                    startVideoService()
                } else {
                    Log.d(TAG, "Camera permission denied: $deniedList")
                    Toast.makeText(this, "Camera permission is required to stream video", Toast.LENGTH_SHORT).show()
                }
            }
    }
    
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            android.R.id.home -> {
                onBackPressedDispatcher.onBackPressed()
                return true
            }
        }
        return super.onOptionsItemSelected(item)
    }
    
    override fun onPause() {
        super.onPause()
        
        // To prevent memory leak
        videoService?.setServerStateListener(null)
    }
}
